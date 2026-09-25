"""Données de référence (échantillon de X_test du P6) et métriques de référence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path("data")


def load_reference(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or DATA_DIR / "reference_sample.parquet")


def load_reference_metrics(path: Path | None = None) -> dict[str, Any]:
    with (path or DATA_DIR / "reference_metrics.json").open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data
