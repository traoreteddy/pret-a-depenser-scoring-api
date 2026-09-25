"""Coût métier et calibrage du seuil de décision (même fonction de coût que le P6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN, COST_FP = 10.0, 1.0


def business_cost(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:  # type: ignore[type-arg]
    pred = proba >= threshold
    fn = np.sum(~pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return float((cost_fn * fn + cost_fp * fp) / len(y))


def cost_curve(
    y: np.ndarray,  # type: ignore[type-arg]
    proba: np.ndarray,  # type: ignore[type-arg]
    thresholds: np.ndarray | None = None,  # type: ignore[type-arg]
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> pd.DataFrame:
    """Coût, taux de refus, rappel et précision pour chaque seuil."""
    ths = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    pos = max(int(y.sum()), 1)
    for t in ths:
        pred = proba >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        rows.append(
            {
                "seuil": float(t),
                "cout": (cost_fn * fn + cost_fp * fp) / len(y),
                "taux_refus": float(pred.mean()),
                "rappel": tp / pos,
                "precision": tp / max(tp + fp, 1),
                "fn": fn,
                "fp": fp,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> tuple[float, float]:
    i = int(curve["cout"].idxmin())
    return float(curve.loc[i, "seuil"]), float(curve.loc[i, "cout"])
