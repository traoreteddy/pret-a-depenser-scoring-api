"""Sélection et chargement du backend de prédiction (une seule fois au démarrage)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scoring_api.config import Settings
from scoring_api.features.spec import FEATURES
from scoring_api.model.base import ModelInfo, Predictor


class FakePredictor:
    """Backend déterministe pour les tests : proba = 0,2 ; 0,9 si AMT_ANNUITY > 100 000."""

    def __init__(self, threshold: float = 0.48) -> None:
        self.info = ModelInfo(name="fake", version="test", backend="fake", threshold=threshold)
        self.features = FEATURES

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        annuity = x[:, FEATURES.index("AMT_ANNUITY")]
        return np.where(np.nan_to_num(annuity) > 100_000, 0.9, 0.2).astype(np.float64)


def load_predictor(settings: Settings) -> Predictor:
    backend = settings.model_backend
    model_dir = settings.model_path
    if backend == "fake":
        return FakePredictor(settings.model_threshold or 0.48)
    if backend == "pyfunc":
        from scoring_api.model.pyfunc_backend import PyfuncPredictor  # noqa: PLC0415

        return PyfuncPredictor(model_dir, settings.model_threshold)
    if backend == "xgb_native":
        from scoring_api.model.xgb_backend import XgbNativePredictor  # noqa: PLC0415

        return XgbNativePredictor(model_dir, settings.model_threshold, settings.xgb_nthread)
    if backend == "onnx":
        from scoring_api.model.onnx_backend import OnnxPredictor  # noqa: PLC0415

        return OnnxPredictor(model_dir, settings.model_threshold, max(settings.xgb_nthread, 1))
    msg = f"Backend inconnu : {backend}"
    raise ValueError(msg)


def warmup(predictor: Predictor, model_dir: Path, rows: int) -> float:
    """Exécute quelques prédictions sur l'exemple MLflow pour amorcer caches et threads."""
    if rows <= 0:
        return 0.0
    example = model_dir / "input_example.json"
    if example.exists():
        with example.open(encoding="utf-8") as f:
            ex = json.load(f)
        cols = ex["columns"]
        idx = [cols.index(c) for c in FEATURES]
        data = np.array(ex["data"], dtype=np.float64)[:, idx]
    else:
        data = np.zeros((1, len(FEATURES)), dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(rows):
        predictor.predict_proba(data[:1])
    return (time.perf_counter() - t0) * 1000
