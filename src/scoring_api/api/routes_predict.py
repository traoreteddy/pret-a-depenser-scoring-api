"""Endpoint de scoring : données brutes du client → probabilité de défaut et décision."""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from scoring_api.api.deps import get_prediction_logger, get_predictor
from scoring_api.api.errors import ModelInferenceError
from scoring_api.api.schemas import ClientInput, ErrorResponse, PredictionResponse
from scoring_api.features.engineering import build_features
from scoring_api.model.base import Predictor, decide
from scoring_api.observability import metrics

router = APIRouter(tags=["scoring"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score de défaut d'un client",
    description=(
        "Reçoit les 32 champs bruts d'un dossier, calcule les 9 variables dérivées, "
        "renvoie la probabilité de défaut et la décision au seuil métier (refus si proba ≥ seuil)."
    ),
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Données invalides (champ manquant, hors plage, mauvais type)",
        },
        500: {"model": ErrorResponse, "description": "Erreur d'inférence"},
        503: {"model": ErrorResponse, "description": "Modèle non chargé"},
    },
)
def predict(
    payload: ClientInput,
    request: Request,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    plogger: Annotated[Any, Depends(get_prediction_logger)],
) -> PredictionResponse:
    # Endpoint synchrone : FastAPI l'exécute dans un pool de threads, l'inférence (CPU) ne bloque
    # donc pas la boucle d'événements.
    request_id = str(request.state.request_id)
    raw = payload.model_dump()

    t0 = time.perf_counter()
    x = build_features(raw)
    t1 = time.perf_counter()
    try:
        proba = float(predictor.predict_proba(x)[0])
    except Exception as exc:
        metrics.prediction_errors.labels(error_type="model_error").inc()
        raise ModelInferenceError(str(exc)) from exc
    t2 = time.perf_counter()

    threshold = predictor.info.threshold
    decision = decide(proba, threshold)
    fe_ms, inf_ms = (t1 - t0) * 1000, (t2 - t1) * 1000

    metrics.observe_prediction(predictor.info.backend, proba, decision, fe_ms, inf_ms)
    if plogger is not None:
        plogger.record(
            request_id=request_id,
            info=predictor.info,
            proba=proba,
            decision=decision,
            latency_features_ms=fe_ms,
            latency_inference_ms=inf_ms,
            raw_input=raw,
            features=x[0],
            client_ref=getattr(request.state, "client_ref", None),
            scenario=getattr(request.state, "scenario", None),
        )

    return PredictionResponse(
        request_id=request_id,
        proba_defaut=proba,
        decision=decision,  # type: ignore[arg-type]
        threshold=threshold,
        model_name=predictor.info.name,
        model_version=predictor.info.version,
        model_backend=predictor.info.backend,
        latency_ms=round(fe_ms + inf_ms, 3),
    )
