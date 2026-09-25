"""Estimation de la performance en production, avec ou sans labels.

Sans labels (CBPE simplifié) : si le modèle est calibré, la probabilité prédite p est l'espérance
du défaut. Pour un seuil t : FN attendus = Σ_{p<t} p, FP attendus = Σ_{p≥t} (1−p), etc. On en
déduit un coût métier et une AUC « attendus ». La calibration est vérifiée sur la référence
(courbe de fiabilité) ; un calibrateur isotonique peut corriger p avant l'estimation.

Avec labels différés : métriques réalisées (AUC, coût, rappel) comparées à l'estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from scoring_api.monitoring.threshold import COST_FN, COST_FP, business_cost


def reliability_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:  # type: ignore[type-arg]
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(proba, dtype=float)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), proba_moyenne=("p", "mean"), taux_observe=("y", "mean")
    )
    return g.reset_index()


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:  # type: ignore[type-arg]
    t = reliability_table(y, proba, bins)
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["proba_moyenne"] - t["taux_observe"])))


def fit_calibrator(y: np.ndarray, proba: np.ndarray) -> IsotonicRegression:  # type: ignore[type-arg]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(proba, dtype=float), np.asarray(y).astype(int))
    return iso


def expected_metrics(
    proba: np.ndarray,
    threshold: float,
    calibrator: IsotonicRegression | None = None,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, Any]:  # type: ignore[type-arg]
    """Métriques attendues à partir des seules probabilités (pas de labels)."""
    p = np.asarray(proba, dtype=float)
    q = calibrator.predict(p) if calibrator is not None else p  # probabilité calibrée
    refused = p >= threshold
    exp_fn = float(q[~refused].sum())
    exp_fp = float((1 - q[refused]).sum())
    exp_tp = float(q[refused].sum())
    exp_tn = float((1 - q[~refused]).sum())
    n = len(p)
    # AUC attendue : TPR/FPR attendus sur une grille de seuils (CBPE)
    grid = np.linspace(0, 1, 101)
    pos, neg = q.sum(), (1 - q).sum()
    tpr = np.array([q[p >= t].sum() / max(pos, 1e-9) for t in grid])
    fpr = np.array([(1 - q[p >= t]).sum() / max(neg, 1e-9) for t in grid])
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    return {
        "n": n,
        "cout_attendu": (cost_fn * exp_fn + cost_fp * exp_fp) / max(n, 1),
        "auc_attendue": auc,
        "taux_defaut_attendu": float(q.mean()),
        "taux_refus": float(refused.mean()),
        "rappel_attendu": exp_tp / max(exp_tp + exp_fn, 1e-9),
        "precision_attendue": exp_tp / max(exp_tp + exp_fp, 1e-9),
        "fn_attendus": exp_fn,
        "fp_attendus": exp_fp,
        "tn_attendus": exp_tn,
        "tp_attendus": exp_tp,
    }


def realized_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, Any]:  # type: ignore[type-arg]
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return {
        "n": len(y),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "cout": business_cost(y, p, threshold, cost_fn, cost_fp),
        "taux_defaut": float(y.mean()),
        "taux_refus": float(pred.mean()),
        "rappel": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "ece": expected_calibration_error(y, p),
    }
