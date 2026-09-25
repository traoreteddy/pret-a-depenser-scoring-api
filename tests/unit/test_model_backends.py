"""Backends de prédiction : interface commune, seuil, non-régression sur l'exemple MLflow."""

import json
from pathlib import Path

import numpy as np
import pytest

from scoring_api.config import Settings
from scoring_api.features.spec import FEATURES
from scoring_api.model.base import ACCEPT, REFUSE, Predictor, decide
from scoring_api.model.registry import FakePredictor, load_predictor, warmup

MODEL_DIR = Path("models/scoring_credit_v1")


def _settings(backend: str) -> Settings:
    return Settings(_env_file=None, model_backend=backend, db_enabled=False)  # type: ignore[call-arg]


def _example() -> np.ndarray:  # type: ignore[type-arg]
    with (MODEL_DIR / "input_example.json").open() as f:
        ex = json.load(f)
    return np.array(ex["data"], dtype=np.float64)[:, [ex["columns"].index(c) for c in FEATURES]]


# Probabilités du modèle P6 sur input_example.json (vérifiées avec mlflow.pyfunc au P6)
EXPECTED = np.array([0.302235, 0.428427, 0.678636, 0.228789, 0.422334])


def test_decide_rule() -> None:
    assert decide(0.48, 0.48) == REFUSE
    assert decide(0.4799, 0.48) == ACCEPT


def test_fake_predictor_is_a_predictor() -> None:
    p = load_predictor(_settings("fake"))
    assert isinstance(p, Predictor) and isinstance(p, FakePredictor)
    assert p.info.backend == "fake" and p.features == FEATURES
    assert warmup(p, MODEL_DIR, 3) >= 0


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Backend inconnu"):
        load_predictor(
            Settings(_env_file=None, model_backend="fake", db_enabled=False).model_copy(
                update={"model_backend": "nope"}
            )
        )  # type: ignore[call-arg]


@pytest.mark.skipif(not (MODEL_DIR / "booster.ubj").exists(), reason="booster.ubj absent")
def test_xgb_native_matches_reference() -> None:
    p = load_predictor(_settings("xgb_native"))
    assert p.info.backend == "xgb_native" and p.info.threshold == pytest.approx(0.48)
    np.testing.assert_allclose(p.predict_proba(_example()), EXPECTED, atol=1e-6)


@pytest.mark.skipif(not (MODEL_DIR / "model.onnx").exists(), reason="model.onnx absent")
def test_onnx_matches_reference() -> None:
    pytest.importorskip("onnxruntime")
    p = load_predictor(_settings("onnx"))
    assert p.info.backend == "onnx"
    np.testing.assert_allclose(p.predict_proba(_example()), EXPECTED, atol=1e-5)


@pytest.mark.slow
def test_pyfunc_matches_reference() -> None:
    p = load_predictor(_settings("pyfunc"))
    np.testing.assert_allclose(p.predict_proba(_example()), EXPECTED, atol=1e-6)
    np.testing.assert_allclose(
        p.predict_pyfunc(_example())["proba_defaut"].to_numpy(), EXPECTED, atol=1e-6
    )  # type: ignore[attr-defined]
