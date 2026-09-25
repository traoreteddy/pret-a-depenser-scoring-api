"""Interface commune aux backends de prédiction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

REFUSE = "Refusé"
ACCEPT = "Accordé"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    version: str
    backend: str
    threshold: float
    run_id: str | None = None


@runtime_checkable
class Predictor(Protocol):
    """Un backend prend une matrice (n, 38) float64 et renvoie la probabilité de défaut (n,)."""

    info: ModelInfo
    features: tuple[str, ...]

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


def decide(proba: float, threshold: float) -> str:
    """Règle métier du P6 : refus si proba >= seuil."""
    return REFUSE if proba >= threshold else ACCEPT
