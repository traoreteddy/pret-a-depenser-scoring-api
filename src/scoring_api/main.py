"""Point d'entrée de l'API de scoring « Prêt à Dépenser »."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scoring_api import __version__
from scoring_api.api import routes_health, routes_metrics, routes_predict
from scoring_api.api.errors import register_error_handlers
from scoring_api.api.middleware import RequestContextMiddleware
from scoring_api.config import Settings, get_settings
from scoring_api.model.registry import load_predictor, warmup
from scoring_api.observability import metrics
from scoring_api.observability.logging import configure_logging, get_logger

log = get_logger("scoring_api")

DESCRIPTION = """
API de scoring crédit de **Prêt à Dépenser**. Le modèle (XGBoost, versionné avec MLflow) est chargé
**une seule fois** au démarrage et réutilisé pour toutes les requêtes.

- `POST /predict` : score de défaut et décision d'un dossier client.
- `GET /health/live`, `GET /health/ready` : sondes de vivacité / disponibilité.
- `GET /metrics` : métriques Prometheus (latence, erreurs, distribution des scores).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    t0 = time.perf_counter()
    predictor = load_predictor(settings)
    load_ms = (time.perf_counter() - t0) * 1000
    warm_ms = warmup(predictor, settings.model_path, settings.warmup_rows)
    app.state.predictor = predictor
    metrics.set_model_info(predictor.info)
    log.info(
        "model_loaded",
        backend=predictor.info.backend,
        name=predictor.info.name,
        version=predictor.info.version,
        threshold=predictor.info.threshold,
        load_ms=round(load_ms, 1),
        warmup_ms=round(warm_ms, 1),
    )

    if settings.db_active:
        from scoring_api.storage.logger import PredictionLogger  # noqa: PLC0415

        plogger = PredictionLogger(settings)
        await plogger.start()
        app.state.prediction_logger = plogger
        app.state.on_validation_error = plogger.record_validation_error

    metrics.app_ready.set(1)
    try:
        yield
    finally:
        metrics.app_ready.set(0)
        running = getattr(app.state, "prediction_logger", None)
        if running is not None:
            await running.stop()
        log.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    app = FastAPI(
        title="Prêt à Dépenser — API de scoring crédit",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "Data Science — Prêt à Dépenser"},
    )
    app.state.settings = settings
    app.state.predictor = None
    app.state.prediction_logger = None
    app.include_router(routes_health.router)
    app.include_router(routes_predict.router)
    if settings.metrics_enabled:
        app.include_router(routes_metrics.router)
    register_error_handlers(app)
    app.add_middleware(
        RequestContextMiddleware,
        on_response=metrics.on_response if settings.metrics_enabled else None,
    )
    return app


app = create_app()
