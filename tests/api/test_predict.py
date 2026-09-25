"""Cas critiques du brief : nominal, champs manquants, hors plage, mauvais types."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scoring_api.main import create_app
from tests.conftest import make_settings


def test_nominal(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["proba_defaut"] <= 1
    assert body["decision"] in {"Accordé", "Refusé"}
    assert body["decision"] == (
        "Refusé" if body["proba_defaut"] >= body["threshold"] else "Accordé"
    )
    assert body["model_backend"] == "fake"
    assert body["request_id"] == r.headers["x-request-id"]
    assert float(r.headers["x-process-time-ms"]) >= 0


def test_request_id_propagated(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/predict", json=payload, headers={"X-Request-ID": "abc-123"})
    assert r.json()["request_id"] == "abc-123"
    assert r.headers["x-request-id"] == "abc-123"


def test_decision_refused_when_high_risk(client: TestClient, payload: dict[str, object]) -> None:
    payload["AMT_ANNUITY"] = 150_000.0  # le prédicteur factice renvoie 0,9
    r = client.post("/predict", json=payload)
    assert r.json()["decision"] == "Refusé"


def test_missing_required_field(client: TestClient, payload: dict[str, object]) -> None:
    del payload["AMT_CREDIT"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert body["details"][0]["loc"] == ["body", "AMT_CREDIT"]
    assert body["details"][0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field", "value", "err_type"),
    [
        ("DAYS_BIRTH", 1826, "less_than_equal"),  # âge -5 ans
        ("DAYS_BIRTH", -55_000, "greater_than_equal"),  # âge 150 ans
        ("AMT_CREDIT", 0, "greater_than"),  # crédit nul
        ("AMT_CREDIT", -1000.0, "greater_than"),
        ("EXT_SOURCE_3", 1.5, "less_than_equal"),
        ("OWN_CAR_AGE", 200.0, "less_than_equal"),
    ],
)
def test_out_of_range(
    client: TestClient, payload: dict[str, object], field: str, value: object, err_type: str
) -> None:
    payload[field] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    d = r.json()["details"][0]
    assert d["loc"] == ["body", field] and d["type"] == err_type


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AMT_CREDIT", "abc"),
        ("AMT_CREDIT", "30000"),
        ("DAYS_BIRTH", "-14000"),
        ("DAYS_BIRTH", -14000.7),
        ("FLAG_DOCUMENT_3", "oui"),
        ("FLAG_DOCUMENT_3", True),
        ("EXT_SOURCE_2", {"a": 1}),
    ],
)
def test_wrong_types(
    client: TestClient, payload: dict[str, object], field: str, value: object
) -> None:
    payload[field] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["details"][0]["loc"] == ["body", field]


def test_unknown_field(client: TestClient, payload: dict[str, object]) -> None:
    payload["AMT_INCOME_TOTAL"] = 0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["details"][0]["type"] == "extra_forbidden"


def test_null_optional_ok(client: TestClient, payload: dict[str, object]) -> None:
    payload["EXT_SOURCE_1"] = None
    payload["EXT_SOURCE_2"] = None
    payload["EXT_SOURCE_3"] = None
    assert client.post("/predict", json=payload).status_code == 200


def test_empty_and_malformed_body(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422
    r = client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_model_error_returns_500_without_trace(
    client: TestClient, payload: dict[str, object]
) -> None:
    class Boom:
        info = client.app.state.predictor.info  # type: ignore[attr-defined]
        features = client.app.state.predictor.features  # type: ignore[attr-defined]

        def predict_proba(self, x: object) -> object:
            raise RuntimeError("secret interne")

    client.app.state.predictor = Boom()  # type: ignore[attr-defined]
    r = client.post("/predict", json=payload)
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "model_error"
    assert "secret interne" not in r.text
    assert "Traceback" not in r.text


def test_predict_503_when_model_not_loaded(
    client_not_started: TestClient, payload: dict[str, object]
) -> None:
    r = client_not_started.post("/predict", json=payload)
    assert r.status_code == 503
    assert r.json()["error"] == "not_ready"


def test_unknown_route_json_error(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"] == "http_error"


def test_threadpool_mode(payload: dict[str, object]) -> None:
    with TestClient(create_app(make_settings(predict_in_threadpool=True))) as c:
        r = c.post("/predict", json=payload)
        assert r.status_code == 200 and r.json()["decision"] in {"Accordé", "Refusé"}
