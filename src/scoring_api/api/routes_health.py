"""Sondes de vivacité et de disponibilité (Kubernetes/Docker-ready)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from scoring_api import __version__
from scoring_api.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["santé"])


@router.get("/live", response_model=HealthResponse, summary="Vivacité : le processus répond")
def live(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", model_loaded=request.app.state.predictor is not None, version=__version__
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Disponibilité : modèle chargé (et base joignable si exigée)",
    responses={503: {"description": "Modèle non chargé ou base indisponible"}},
)
async def ready(request: Request, response: Response) -> HealthResponse:
    state = request.app.state
    predictor = getattr(state, "predictor", None)
    settings = state.settings
    logger = getattr(state, "prediction_logger", None)

    db_status: str = "disabled"
    if settings.db_active:
        db_status = "unknown" if logger is None else ("ok" if await logger.ping() else "degraded")

    if predictor is None or (settings.ready_require_db and db_status != "ok"):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            model_loaded=predictor is not None,
            db=db_status,
            version=__version__,
        )  # type: ignore[arg-type]

    info = predictor.info
    return HealthResponse(
        status="ok" if db_status in {"ok", "disabled"} else "degraded",
        model_loaded=True,
        model={
            "name": info.name,
            "version": info.version,
            "backend": info.backend,
            "threshold": info.threshold,
        },
        db=db_status,  # type: ignore[arg-type]
        version=__version__,
    )
