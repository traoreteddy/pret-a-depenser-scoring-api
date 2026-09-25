from fastapi.testclient import TestClient


def test_live_ok(client: TestClient) -> None:
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "x-request-id" in r.headers


def test_ready_ok_after_startup(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model"]["backend"] == "fake"
    assert body["db"] == "disabled"


def test_ready_503_before_model_loaded(client_not_started: TestClient) -> None:
    r = client_not_started.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"


def test_docs_and_openapi(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    props = spec["components"]["schemas"]["ClientInput"]["properties"]
    assert len(props) == 32
