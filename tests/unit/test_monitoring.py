import numpy as np
import pandas as pd
import pytest

from scoring_api.monitoring.drift import (
    dataset_summary,
    drift_table,
    psi,
    psi_status,
    refusal_rate_ci,
)
from scoring_api.monitoring.performance import expected_metrics, fit_calibrator, realized_metrics
from scoring_api.monitoring.threshold import business_cost, cost_curve, optimal_threshold

rng = np.random.default_rng(0)


def test_psi_zero_for_same_distribution() -> None:
    a = pd.Series(rng.normal(size=5000))
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi_status(psi(a, pd.Series(rng.normal(size=5000)))) == "stable"


def test_psi_detects_shift_and_missing() -> None:
    a = pd.Series(rng.normal(size=5000))
    assert psi(a, a + 1.0) > 0.25
    b = a.copy()
    b.iloc[:2500] = np.nan
    assert psi(a, b) > 0.25


def test_drift_table_and_summary() -> None:
    ref = pd.DataFrame(
        {
            "x": rng.normal(size=2000),
            "y": rng.normal(size=2000),
            "proba_defaut": rng.uniform(size=2000),
        }
    )
    cur = ref.copy()
    cur["x"] += 0.8
    table = drift_table(ref, cur, columns=("x", "y", "proba_defaut"))
    assert table.iloc[0]["colonne"] == "x" and table.iloc[0]["statut_psi"] == "dérive"
    assert bool(table.set_index("colonne").loc["y", "ks_derive"]) is False
    s = dataset_summary(table, len(cur))
    assert s["n_psi_drift"] == 1 and s["dataset_drift"] is True  # 1/3 > 30 %


def test_refusal_ci() -> None:
    p, lo, hi = refusal_rate_ci(320, 1000)
    assert lo < p < hi and p == 0.32


def test_cost_curve_matches_p6_cost_function() -> None:
    y = np.array([1, 0, 0, 1, 0])
    proba = np.array([0.9, 0.2, 0.6, 0.3, 0.1])
    # seuil 0,5 : FN = 1 (0.3), FP = 1 (0.6) → (10 + 1) / 5
    assert business_cost(y, proba, 0.5) == pytest.approx(11 / 5)
    curve = cost_curve(y, proba)
    t, c = optimal_threshold(curve)
    assert c <= curve["cout"].min() + 1e-12 and 0 < t < 1


def test_expected_vs_realized_on_calibrated_data() -> None:
    p = rng.uniform(0.02, 0.9, size=20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # données parfaitement calibrées
    exp = expected_metrics(p, 0.48)
    real = realized_metrics(y, p, 0.48)
    assert exp["cout_attendu"] == pytest.approx(real["cout"], rel=0.05)
    assert exp["auc_attendue"] == pytest.approx(real["auc"], abs=0.02)
    cal = fit_calibrator(y, p)
    assert cal.predict(np.array([0.5]))[0] == pytest.approx(0.5, abs=0.05)
