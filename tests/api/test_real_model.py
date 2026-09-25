"""Non-régression avec le vrai modèle MLflow (marqué slow)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scoring_api.features.spec import FEATURES, RAW_FIELDS
from tests.conftest import MODEL_DIR

pytestmark = pytest.mark.slow


def test_ready_reports_real_model(real_client: TestClient) -> None:
    body = real_client.get("/health/ready").json()
    assert body["model"]["name"] == "scoring_credit"
    assert body["model"]["backend"] == "pyfunc"
    assert body["model"]["threshold"] == pytest.approx(0.48)


def test_predictions_match_input_example(real_client: TestClient) -> None:
    """Les 5 lignes de l'exemple MLflow doivent redonner exactement les probabilités du P6."""
    with (MODEL_DIR / "input_example.json").open() as f:
        ex = json.load(f)
    cols = ex["columns"]
    predictor = real_client.app.state.predictor  # type: ignore[attr-defined]
    for row in ex["data"]:
        feat = dict(zip(cols, row, strict=True))
        raw = {k: feat[k] for k in RAW_FIELDS if k in feat}
        raw["DAYS_BIRTH"] = round(-feat["AGE_ANNEES"] * 365.25)
        raw["AMT_CREDIT"] = feat["CREDIT_ANNUITY_RATIO"] * (feat["AMT_ANNUITY"] + 1e-6)
        raw["PREV_REFUSED_COUNT"] = feat["PREV_REFUSED_RATIO"] * (feat["PREV_COUNT"] + 1e-6)
        raw = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in raw.items()}
        for k in ("DAYS_ID_PUBLISH", "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE"):
            raw[k] = int(raw[k])
        r = real_client.post("/predict", json=raw)
        assert r.status_code == 200, r.text
        expected = predictor.predict_pyfunc(np.array([[feat[c] for c in FEATURES]], dtype=float))
        assert r.json()["proba_defaut"] == pytest.approx(
            float(expected["proba_defaut"][0]), abs=1e-4
        )
        assert r.json()["decision"] == expected["decision"][0]
