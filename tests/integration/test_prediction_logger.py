"""Journalisation de bout en bout dans PostgreSQL (nécessite DATABASE_URL)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient

from scoring_api.api.schemas import EXAMPLE_CLIENT
from scoring_api.main import create_app
from tests.conftest import make_settings

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def db_client() -> Iterator[TestClient]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL non défini")
    if "test" not in DATABASE_URL.rsplit("/", 1)[-1]:
        # Garde-fou : le fixture supprime les tables, il ne doit jamais viser une base de production.
        pytest.skip("DATABASE_URL doit pointer vers une base dont le nom contient « test »")
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("DROP TABLE IF EXISTS prediction_labels, api_errors, predictions CASCADE")
        conn.commit()
    settings = make_settings(
        db_enabled=True,
        database_url=DATABASE_URL,
        log_sample_rate=0.25,
        log_batch_size=50,
        log_flush_interval_s=0.2,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _wait_rows(table: str, expected: int, timeout: float = 5.0) -> int:
    assert DATABASE_URL
    deadline = time.monotonic() + timeout
    n = 0
    while time.monotonic() < deadline:
        with psycopg.connect(DATABASE_URL) as conn:
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # type: ignore[index]
        if n >= expected:
            break
        time.sleep(0.1)
    return n


def test_ready_reports_db_ok(db_client: TestClient) -> None:
    assert db_client.get("/health/ready").json()["db"] == "ok"


def test_predictions_are_persisted_with_sampling(db_client: TestClient) -> None:
    for i in range(300):
        r = db_client.post("/predict", json=EXAMPLE_CLIENT, headers={"X-Client-Ref": f"c{i}"})
        assert r.status_code == 200
    assert _wait_rows("predictions", 300) == 300
    with psycopg.connect(DATABASE_URL) as conn:  # type: ignore[arg-type]
        sampled, with_input, refs = conn.execute(
            "SELECT sum(sampled::int), count(raw_input), count(DISTINCT client_ref) FROM predictions"
        ).fetchone()  # type: ignore[misc]
    assert with_input == sampled
    assert 45 <= sampled <= 105  # 25 % ± marge sur 300 tirages
    assert refs == 300


def test_scenario_header_forces_full_logging(db_client: TestClient) -> None:
    db_client.post(
        "/predict",
        json=EXAMPLE_CLIENT,
        headers={"X-Scenario": "normal", "X-Request-ID": "11111111-1111-1111-1111-111111111111"},
    )
    _wait_rows("predictions", 301)
    with psycopg.connect(DATABASE_URL) as conn:  # type: ignore[arg-type]
        row = conn.execute(
            "SELECT sampled, scenario, features->>'AGE_ANNEES' FROM predictions WHERE request_id = %s",
            ("11111111-1111-1111-1111-111111111111",),
        ).fetchone()
    assert row is not None and row[0] is True and row[1] == "normal" and row[2] is not None


def test_validation_errors_are_persisted(db_client: TestClient) -> None:
    db_client.post("/predict", json={**EXAMPLE_CLIENT, "DAYS_BIRTH": 1826})
    db_client.post("/predict", json={**EXAMPLE_CLIENT, "AMT_CREDIT": "abc"})
    assert _wait_rows("api_errors", 2) == 2
    with psycopg.connect(DATABASE_URL) as conn:  # type: ignore[arg-type]
        rows = conn.execute(
            "SELECT error_type, detail->0->>'loc', raw_body->>'DAYS_BIRTH' FROM api_errors ORDER BY id"
        ).fetchall()
    assert rows[0][0] == "validation_error" and "DAYS_BIRTH" in rows[0][1] and rows[0][2] == "1826"
