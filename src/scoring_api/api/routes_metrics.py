"""Exposition Prometheus."""

from __future__ import annotations

from fastapi import APIRouter, Response

from scoring_api.observability import metrics

router = APIRouter(tags=["observabilité"])


@router.get("/metrics", summary="Métriques Prometheus", include_in_schema=True)
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
