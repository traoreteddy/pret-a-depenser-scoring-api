"""Copie les artefacts MLflow du modèle champion P6 dans `models/scoring_credit_v1/`.

Usage : uv run python scripts/import_model.py --source <dossier artifacts MLflow>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

FILES = (
    "MLmodel",
    "python_model.pkl",
    "input_example.json",
    "serving_input_example.json",
    "requirements.txt",
    "conda.yaml",
    "python_env.yaml",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", default=Path("models/scoring_credit_v1"), type=Path)
    parser.add_argument("--name", default="scoring_credit")
    parser.add_argument("--version", default="1")
    args = parser.parse_args()

    src: Path = args.source
    dest: Path = args.dest
    if not (src / "MLmodel").exists():
        print(f"MLmodel introuvable dans {src}", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
            print(f"copié  {name}")

    with (dest / "MLmodel").open(encoding="utf-8") as f:
        mlmodel = yaml.safe_load(f)
    inputs = json.loads(mlmodel["signature"]["inputs"])

    # Vérification : le modèle se charge et prédit sur l'exemple
    import mlflow.pyfunc  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    model = mlflow.pyfunc.load_model(str(dest))
    with (dest / "input_example.json").open(encoding="utf-8") as f:
        ex = json.load(f)
    out = model.predict(pd.DataFrame(ex["data"], columns=ex["columns"]))
    inner = model.unwrap_python_model()
    threshold = float(inner.seuil)
    print(out)

    meta = {
        "name": args.name,
        "version": args.version,
        "run_id": mlmodel.get("run_id"),
        "model_id": mlmodel.get("model_id"),
        "mlflow_version": mlmodel.get("mlflow_version"),
        "threshold": threshold,
        "features": [c["name"] for c in inputs],
        "n_features": len(inputs),
        "sha256_python_model_pkl": sha256(dest / "python_model.pkl"),
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(src),
    }
    with (dest / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"model_meta.json écrit (seuil={threshold}, {len(inputs)} variables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
