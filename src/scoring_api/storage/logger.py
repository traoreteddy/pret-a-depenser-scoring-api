"""Journal des prédictions : file asynchrone bornée + insertion par lots, sans bloquer /predict.

Le endpoint appelle `record()` (quelques microsecondes : construction du dictionnaire et mise en
file). Une tâche de fond vide la file par lots et écrit dans PostgreSQL. Si la base est lente
ou indisponible, la file absorbe, puis abandonne (compteur `prediction_log_dropped_total`) :
l'API continue de servir.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import Request
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from scoring_api.api.schemas import ErrorDetail
from scoring_api.config import Settings
from scoring_api.features.spec import FEATURES
from scoring_api.model.base import ModelInfo
from scoring_api.observability import metrics
from scoring_api.observability.logging import get_logger
from scoring_api.storage import db
from scoring_api.storage.sampling import should_sample

log = get_logger("scoring_api.storage")

INSERT_PREDICTION = """
INSERT INTO predictions (request_id, ts, model_name, model_version, model_backend, threshold,
    proba_defaut, decision, latency_total_ms, latency_features_ms, latency_inference_ms,
    status_code, error_type, error_message, sampled, raw_input, features, client_ref, scenario)
VALUES (%(request_id)s, %(ts)s, %(model_name)s, %(model_version)s, %(model_backend)s, %(threshold)s,
    %(proba_defaut)s, %(decision)s, %(latency_total_ms)s, %(latency_features_ms)s, %(latency_inference_ms)s,
    %(status_code)s, %(error_type)s, %(error_message)s, %(sampled)s, %(raw_input)s, %(features)s,
    %(client_ref)s, %(scenario)s)
ON CONFLICT (request_id) DO NOTHING
"""

INSERT_ERROR = """
INSERT INTO api_errors (request_id, ts, path, status_code, error_type, detail, raw_body, scenario)
VALUES (%(request_id)s, %(ts)s, %(path)s, %(status_code)s, %(error_type)s, %(detail)s, %(raw_body)s, %(scenario)s)
"""


def _clean(value: Any) -> Any:
    """NaN → None pour JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class PredictionLogger:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(
            maxsize=settings.log_queue_maxsize
        )
        self.pool: AsyncConnectionPool | None = None
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False
        self.inserted = 0

    # ------------------------------------------------------------------ cycle de vie
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.pool = db.make_pool(self.settings)
        try:
            await self.pool.open(wait=True, timeout=10)
            if self.settings.db_auto_migrate:
                await db.ensure_schema(self.pool)
            log.info("prediction_logger_started", pool_max=self.settings.db_pool_max)
        except Exception as exc:
            # Base injoignable au démarrage : on démarre quand même, le consommateur réessaiera.
            log.error("prediction_logger_db_unavailable", error=str(exc))
            metrics.log_db_errors.inc()
        self._task = asyncio.create_task(self._run(), name="prediction-logger")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        if self.pool is not None:
            await self.pool.close()
        log.info("prediction_logger_stopped", inserted=self.inserted)

    async def ping(self) -> bool:
        return self.pool is not None and await db.ping(self.pool)

    # ------------------------------------------------------------------ producteurs
    def _enqueue(self, kind: str, row: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait((kind, row))
        except asyncio.QueueFull:
            metrics.log_dropped.inc()
        metrics.log_queue_size.set(self.queue.qsize())

    def _submit(self, kind: str, row: dict[str, Any]) -> None:
        """Thread-safe : `record()` est appelé depuis le pool de threads de FastAPI."""
        if self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._enqueue, kind, row)

    def record(
        self,
        *,
        request_id: str,
        info: ModelInfo,
        proba: float,
        decision: str,
        latency_features_ms: float,
        latency_inference_ms: float,
        raw_input: dict[str, Any],
        features: np.ndarray[Any, np.dtype[np.float64]],
        client_ref: str | None,
        scenario: str | None,
    ) -> None:
        sampled = should_sample(
            request_id,
            proba,
            threshold=info.threshold,
            rate=self.settings.log_sample_rate,
            grey_zone=self.settings.log_grey_zone,
            forced=scenario is not None,
        )
        row: dict[str, Any] = {
            "request_id": request_id,
            "ts": datetime.now(UTC),
            "model_name": info.name,
            "model_version": info.version,
            "model_backend": info.backend,
            "threshold": info.threshold,
            "proba_defaut": proba,
            "decision": decision,
            "latency_total_ms": latency_features_ms + latency_inference_ms,
            "latency_features_ms": latency_features_ms,
            "latency_inference_ms": latency_inference_ms,
            "status_code": 200,
            "error_type": None,
            "error_message": None,
            "sampled": sampled,
            "raw_input": Jsonb({k: _clean(v) for k, v in raw_input.items()}) if sampled else None,
            "features": Jsonb(
                {f: _clean(float(v)) for f, v in zip(FEATURES, features, strict=True)}
            )
            if sampled
            else None,
            "client_ref": client_ref,
            "scenario": scenario,
        }
        self._submit("prediction", row)

    async def record_validation_error(self, request: Request, details: list[ErrorDetail]) -> None:
        try:
            body = await request.body()
            import json  # noqa: PLC0415

            raw_body: Any = json.loads(body) if body else None
        except Exception:
            raw_body = {"_unparseable": True}
        row = {
            "request_id": str(getattr(request.state, "request_id", "")),
            "ts": datetime.now(UTC),
            "path": request.url.path,
            "status_code": 422,
            "error_type": "validation_error",
            "detail": Jsonb([d.model_dump() for d in details]),
            "raw_body": Jsonb(raw_body) if isinstance(raw_body, dict | list) else None,
            "scenario": getattr(request.state, "scenario", None),
        }
        self._enqueue("error", row)

    # ------------------------------------------------------------------ consommateur
    async def _drain(self) -> list[tuple[str, dict[str, Any]]]:
        """Attend au moins un élément puis vide la file (jusqu'au lot max ou l'intervalle)."""
        batch: list[tuple[str, dict[str, Any]]] = []
        deadline = time.monotonic() + self.settings.log_flush_interval_s
        while len(batch) < self.settings.log_batch_size:
            remaining = deadline - time.monotonic()
            if batch and remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self.queue.get(), timeout=max(remaining, 0.05) if batch else 0.5
                )
            except TimeoutError:
                if batch or self._stopping:
                    break
                continue
            batch.append(item)
        metrics.log_queue_size.set(self.queue.qsize())
        return batch

    async def _flush(self, batch: list[tuple[str, dict[str, Any]]]) -> None:
        if self.pool is None:
            return
        preds = [row for kind, row in batch if kind == "prediction"]
        errs = [row for kind, row in batch if kind == "error"]
        async with self.pool.connection() as conn, conn.cursor() as cur:
            if preds:
                await cur.executemany(INSERT_PREDICTION, preds)
            if errs:
                await cur.executemany(INSERT_ERROR, errs)
            await conn.commit()
        self.inserted += len(batch)
        metrics.log_batches.inc()

    async def _run(self) -> None:
        backoff = 0.5
        while not (self._stopping and self.queue.empty()):
            batch = await self._drain()
            if not batch:
                continue
            try:
                await self._flush(batch)
                backoff = 0.5
            except Exception as exc:
                metrics.log_db_errors.inc()
                log.error("prediction_log_flush_failed", n=len(batch), error=str(exc)[:200])
                # On remet le lot en file si possible, sinon on l'abandonne (compté).
                for item in batch:
                    self._enqueue(*item)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10)
