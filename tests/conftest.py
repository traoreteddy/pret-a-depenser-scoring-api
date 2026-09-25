"""Fixtures partagées : application avec prédicteur factice, client HTTP, modèle réel (slow)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scoring_api.api.schemas import EXAMPLE_CLIENT
from scoring_api.config import Settings
from scoring_api.main import create_app

MODEL_DIR = Path("models/scoring_credit_v1")


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "app_env": "test",
        "model_backend": "fake",
        "db_enabled": False,
        "log_format": "console",
        "log_level": "WARNING",
        "warmup_rows": 0,
        "metrics_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_not_started(settings: Settings) -> TestClient:
    """Client sans lifespan : le modèle n'est pas chargé."""
    return TestClient(create_app(settings))


@pytest.fixture
def payload() -> dict[str, object]:
    return dict(EXAMPLE_CLIENT)


@pytest.fixture(scope="session")
def real_client() -> Iterator[TestClient]:
    if not (MODEL_DIR / "python_model.pkl").exists():
        pytest.skip("modèle réel absent")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    app = create_app(make_settings(model_backend="pyfunc", warmup_rows=2))
    with TestClient(app) as c:
        yield c
