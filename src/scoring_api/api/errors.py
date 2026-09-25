"""Gestion centralisée des erreurs : réponses JSON uniformes, jamais de stack trace au client."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from scoring_api.api.schemas import ErrorDetail, ErrorResponse
from scoring_api.observability.logging import get_logger

log = get_logger("scoring_api.errors")


class ModelNotReadyError(Exception):
    """Le modèle n'est pas (encore) chargé."""


class ModelInferenceError(Exception):
    """Le backend de prédiction a levé une exception."""


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _response(
    request: Request, code: int, error: str, message: str, details: Any = None
) -> JSONResponse:
    body = ErrorResponse(
        request_id=_request_id(request), error=error, message=message, details=details
    )
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))


def _details(exc: RequestValidationError) -> list[ErrorDetail]:
    out: list[ErrorDetail] = []
    for e in exc.errors():
        loc = [x for x in e.get("loc", ()) if isinstance(x, str | int)]
        out.append(ErrorDetail(loc=loc, msg=str(e.get("msg", "")), type=str(e.get("type", ""))))
    return out


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = _details(exc)
        log.warning(
            "validation_error",
            n_errors=len(details),
            first=details[0].model_dump() if details else None,
        )
        hook = getattr(request.app.state, "on_validation_error", None)
        if hook is not None:
            await hook(request, details)
        return _response(
            request,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            "Données d'entrée invalides",
            details,
        )

    @app.exception_handler(ModelNotReadyError)
    async def _not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
        return _response(
            request, status.HTTP_503_SERVICE_UNAVAILABLE, "not_ready", "Modèle non chargé"
        )

    @app.exception_handler(ModelInferenceError)
    async def _model_error(request: Request, exc: ModelInferenceError) -> JSONResponse:
        log.error("model_error", exc_info=exc.__cause__ or exc)
        return _response(
            request, status.HTTP_500_INTERNAL_SERVER_ERROR, "model_error", "Échec de l'inférence"
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _response(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        log.error("internal_error", exc_info=exc)
        return _response(
            request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "Erreur interne"
        )
