"""Dépendances FastAPI (accès à l'état applicatif chargé au démarrage)."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from scoring_api.api.errors import ModelNotReadyError
from scoring_api.model.base import Predictor


def get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ModelNotReadyError
    return predictor


def get_prediction_logger(request: Request) -> Any:
    """Journal des prédictions (None tant que le stockage n'est pas configuré)."""
    return getattr(request.app.state, "prediction_logger", None)
