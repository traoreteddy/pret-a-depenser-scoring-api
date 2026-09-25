from fastapi.testclient import TestClient

from scoring_api.main import create_app
from tests.conftest import make_settings


def test_metrics_exposed(client: TestClient, payload: dict[str, object]) -> None:
    client.post("/predict", json=payload)
    client.post("/predict", json={**payload, "AMT_CREDIT": "x"})
    text = client.get("/metrics").text
    assert "predictions_total{decision=" in text
    assert 'http_request_duration_seconds_bucket{le="0.18",path="/predict"}' in text
    assert 'http_requests_total{method="POST",path="/predict",status="422"}' in text
    assert 'model_info{backend="fake"' in text
    assert "app_ready 1.0" in text


def test_model_error_counter(client: TestClient, payload: dict[str, object]) -> None:
    class Boom:
        info = client.app.state.predictor.info  # type: ignore[attr-defined]
        features = client.app.state.predictor.features  # type: ignore[attr-defined]

        def predict_proba(self, x: object) -> object:
            raise ValueError

    client.app.state.predictor = Boom()  # type: ignore[attr-defined]
    client.post("/predict", json=payload)
    assert 'prediction_errors_total{error_type="model_error"}' in client.get("/metrics").text


def test_metrics_disabled() -> None:
    with TestClient(create_app(make_settings(metrics_enabled=False))) as c:
        assert c.get("/metrics").status_code == 404
