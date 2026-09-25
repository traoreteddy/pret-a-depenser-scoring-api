"""Lecture des métadonnées du modèle (`model_meta.json`, `MLmodel`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_mlmodel(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "MLmodel").open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def signature_inputs(model_dir: Path) -> list[str]:
    """Noms des colonnes d'entrée déclarées dans la signature MLflow, dans l'ordre."""
    mlmodel = read_mlmodel(model_dir)
    inputs = json.loads(mlmodel["signature"]["inputs"])
    return [c["name"] for c in inputs]


def read_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model_meta.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data
