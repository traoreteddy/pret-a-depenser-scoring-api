"""Le port de FeatureEngineer doit reproduire exactement les formules du notebook P6."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scoring_api.api.schemas import EXAMPLE_CLIENT
from scoring_api.features.engineering import build_features, derived_features, engineer_frame
from scoring_api.features.spec import ENGINEERED, FEATURES, INT_FEATURES, RAW_FIELDS
from scoring_api.model.meta import signature_inputs

MODEL_DIR = Path("models/scoring_credit_v1")


def test_spec_coherence() -> None:
    assert len(FEATURES) == 38
    assert len(RAW_FIELDS) == 32
    assert set(ENGINEERED) <= set(FEATURES)
    assert not set(ENGINEERED) & set(RAW_FIELDS)
    assert set(FEATURES) >= INT_FEATURES


def test_spec_matches_mlflow_signature() -> None:
    assert tuple(signature_inputs(MODEL_DIR)) == FEATURES


def test_derived_formulas_by_hand() -> None:
    raw = {
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": None,
        "AMT_CREDIT": 100_000.0,
        "AMT_ANNUITY": 5_000.0,
        "AMT_GOODS_PRICE": 90_000.0,
        "DAYS_BIRTH": -14_610,
        "DAYS_ID_PUBLISH": -1_461,
        "PREV_REFUSED_COUNT": 1.0,
        "PREV_COUNT": 4.0,
    }
    d = derived_features(raw)
    assert d["EXT_SOURCE_MEAN"] == pytest.approx(0.35)
    assert d["EXT_SOURCE_MIN"] == pytest.approx(0.2)
    assert d["EXT_SOURCE_STD"] == pytest.approx(np.std([0.2, 0.5], ddof=1))
    assert d["EXT_SOURCE_NB_NAN"] == 1
    assert d["CREDIT_ANNUITY_RATIO"] == pytest.approx(100_000 / (5_000 + 1e-6))
    assert d["CREDIT_GOODS_RATIO"] == pytest.approx(100_000 / (90_000 + 1e-6))
    assert d["AGE_ANNEES"] == pytest.approx(40.0)
    assert d["ID_PUBLISH_AGE_RATIO"] == pytest.approx(0.1)
    assert d["PREV_REFUSED_RATIO"] == pytest.approx(1 / (4 + 1e-6))


def test_std_is_nan_with_single_ext_source() -> None:
    d = derived_features({**EXAMPLE_CLIENT, "EXT_SOURCE_1": None, "EXT_SOURCE_2": None})
    assert math.isnan(d["EXT_SOURCE_STD"])
    assert d["EXT_SOURCE_NB_NAN"] == 2
    assert d["EXT_SOURCE_MEAN"] == EXAMPLE_CLIENT["EXT_SOURCE_3"]


def test_all_ext_sources_missing() -> None:
    d = derived_features(
        {**EXAMPLE_CLIENT, "EXT_SOURCE_1": None, "EXT_SOURCE_2": None, "EXT_SOURCE_3": None}
    )
    assert math.isnan(d["EXT_SOURCE_MEAN"]) and math.isnan(d["EXT_SOURCE_MIN"])
    assert d["EXT_SOURCE_NB_NAN"] == 3


def test_build_features_shape_order_and_nan() -> None:
    x = build_features(EXAMPLE_CLIENT)
    assert x.shape == (1, 38) and x.dtype == np.float64
    assert x[0, FEATURES.index("AMT_ANNUITY")] == EXAMPLE_CLIENT["AMT_ANNUITY"]
    assert math.isnan(x[0, FEATURES.index("EXT_SOURCE_1")])
    for f in INT_FEATURES:
        assert not math.isnan(x[0, FEATURES.index(f)]), f


def test_matches_mlflow_input_example() -> None:
    """Reconstruit les champs bruts à partir de l'exemple MLflow et retrouve les 38 valeurs."""
    with (MODEL_DIR / "input_example.json").open() as f:
        ex = json.load(f)
    cols = ex["columns"]
    for row in ex["data"]:
        feat = dict(zip(cols, row, strict=True))
        # Champs bruts absents de la signature : on les inverse depuis les dérivées
        age_days = -feat["AGE_ANNEES"] * 365.25
        raw = {k: feat[k] for k in RAW_FIELDS if k in feat}
        raw["DAYS_BIRTH"] = age_days
        raw["AMT_CREDIT"] = feat["CREDIT_ANNUITY_RATIO"] * (feat["AMT_ANNUITY"] + 1e-6)
        raw["PREV_REFUSED_COUNT"] = feat["PREV_REFUSED_RATIO"] * (feat["PREV_COUNT"] + 1e-6)
        x = build_features(raw)[0]
        expected = np.array([feat[c] for c in FEATURES], dtype=np.float64)
        mask = ~np.isnan(expected)
        np.testing.assert_allclose(x[mask], expected[mask], rtol=1e-7, atol=1e-9)
        assert np.isnan(x[~mask]).all()


def test_engineer_frame_equals_row_version() -> None:
    pd = pytest.importorskip("pandas")
    rows = [
        EXAMPLE_CLIENT,
        {**EXAMPLE_CLIENT, "EXT_SOURCE_1": 0.3, "PREV_COUNT": 0.0, "PREV_REFUSED_COUNT": 0.0},
    ]
    df = pd.DataFrame(rows).astype({"DAYS_BIRTH": "int64", "DAYS_ID_PUBLISH": "int64"})
    frame = engineer_frame(df).to_numpy(dtype=np.float64)
    rowwise = np.vstack([build_features(r) for r in rows])
    np.testing.assert_allclose(
        np.nan_to_num(frame, nan=-1), np.nan_to_num(rowwise, nan=-1), rtol=1e-12
    )
