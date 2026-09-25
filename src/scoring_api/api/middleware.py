"""Middleware ASGI léger : identifiant de requête, chronométrage, log d'accès JSON."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scoring_api.observability.logging import get_logger

log = get_logger("scoring_api.access")

REQUEST_ID_HEADER = b"x-request-id"
CLIENT_REF_HEADER = b"x-client-ref"
SCENARIO_HEADER = b"x-scenario"


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k == name:
            return str(v.decode("latin-1"))
    return None


class RequestContextMiddleware:
    """Ajoute `request_id`, `X-Process-Time-Ms` et journalise chaque requête HTTP.

    Implémenté en ASGI pur (pas de BaseHTTPMiddleware) pour un surcoût minimal.
    Les métriques Prometheus sont branchées via `on_response` (voir observability.metrics).
    """

    def __init__(
        self,
        app: ASGIApp,
        on_response: Callable[[str, str, int, float], Awaitable[None] | None] | None = None,
    ) -> None:
        self.app = app
        self.on_response = on_response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _header(scope, REQUEST_ID_HEADER) or str(uuid.uuid4())
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["client_ref"] = _header(scope, CLIENT_REF_HEADER)
        state["scenario"] = _header(scope, SCENARIO_HEADER)
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                headers.append((b"x-process-time-ms", f"{elapsed_ms:.2f}".encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            path = scope.get("path", "")
            method = scope.get("method", "")
            status = status_holder["status"]
            if path not in {"/metrics", "/health/live"}:
                log.info(
                    "http_request",
                    method=method,
                    path=path,
                    status=status,
                    duration_ms=round(elapsed * 1000, 2),
                )
            if self.on_response is not None:
                result = self.on_response(method, path, status, elapsed)
                if result is not None:
                    await result
            structlog.contextvars.unbind_contextvars("request_id")
