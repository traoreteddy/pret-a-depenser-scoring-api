"""Lecture des données de production (PostgreSQL) sous forme de DataFrames."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from scoring_api.features.spec import FEATURES, RAW_FIELDS

PRED_COLUMNS = (
    "request_id, ts, model_name, model_version, model_backend, threshold, proba_defaut, decision, "
    "latency_total_ms, latency_features_ms, latency_inference_ms, status_code, error_type, sampled, "
    "raw_input, features, client_ref, scenario"
)


def _where(
    since: datetime | None, until: datetime | None, scenario: str | None
) -> tuple[str, list[Any]]:
    clauses: list[str] = ["TRUE"]
    params: list[Any] = []
    if since is not None:
        clauses.append("p.ts >= %s")
        params.append(since)
    if until is not None:
        clauses.append("p.ts < %s")
        params.append(until)
    if scenario is not None:
        clauses.append("p.scenario = %s")
        params.append(scenario)
    return " AND ".join(clauses), params


def load_predictions(
    database_url: str,
    since: datetime | None = None,
    until: datetime | None = None,
    scenario: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Toutes les prédictions (ligne légère) + label différé s'il existe."""
    where, params = _where(since, until, scenario)
    sql = f"""
        SELECT {PRED_COLUMNS.replace("request_id", "p.request_id")}, l.target
        FROM predictions p
        LEFT JOIN prediction_labels l ON l.request_id = p.request_id
        WHERE {where}
        ORDER BY p.ts
        {f"LIMIT {int(limit)}" if limit else ""}
    """
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        rows = conn.execute(sql, params).fetchall()  # type: ignore[arg-type]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["request_id"] = df["request_id"].astype(str)
    return df


def expand_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Lignes échantillonnées : déplie raw_input (32 bruts) et features (38 variables) en colonnes."""
    if df.empty:
        return df
    sampled = df[df["sampled"] & df["raw_input"].notna()].copy()
    if sampled.empty:
        return sampled
    raw = pd.DataFrame(list(sampled["raw_input"]), index=sampled.index)[list(RAW_FIELDS)]
    feats = pd.DataFrame(list(sampled["features"]), index=sampled.index)[list(FEATURES)]
    feats = feats[[c for c in FEATURES if c not in RAW_FIELDS]]  # évite les doublons de colonnes
    out = pd.concat([sampled.drop(columns=["raw_input", "features"]), raw, feats], axis=1)
    return out.astype(
        {c: "float64" for c in list(RAW_FIELDS) + list(feats.columns)}, errors="ignore"
    )


def load_errors(database_url: str, since: datetime | None = None, limit: int = 500) -> pd.DataFrame:
    where, params = _where(since, None, None)
    sql = f"SELECT request_id, ts, path, status_code, error_type, detail, scenario FROM api_errors p WHERE {where} ORDER BY ts DESC LIMIT {int(limit)}"
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        rows = conn.execute(sql, params).fetchall()  # type: ignore[arg-type]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def scenario_list(database_url: str) -> list[str]:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT DISTINCT scenario FROM predictions WHERE scenario IS NOT NULL ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows]
