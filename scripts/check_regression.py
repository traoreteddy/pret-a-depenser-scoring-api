"""Non-régression entre backends : mêmes probabilités et mêmes décisions que le pyfunc MLflow.

Compare sur les 5 000 dossiers de data/prod_sample.parquet (features calculées par l'API).
Critères : écart max < 1e-6 (xgb_native), < 1e-4 (onnx, arbres évalués en float32) ; 100 % des
décisions identiques. Résultats : benchmarks/results/regression_<backend>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from scoring_api.config import Settings
from scoring_api.features.engineering import engineer_frame
from scoring_api.model.registry import load_predictor

TOLERANCE = {"xgb_native": 1e-6, "onnx": 1e-4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["xgb_native", "onnx"])
    parser.add_argument("--sample", type=Path, default=Path("data/prod_sample.parquet"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    df = pd.read_parquet(args.sample)
    x = engineer_frame(df).to_numpy(dtype=np.float64)
    base = load_predictor(Settings(_env_file=None, model_backend="pyfunc", db_enabled=False))  # type: ignore[call-arg]
    threshold = base.info.threshold
    ref = base.predict_proba(x)
    ok = True
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.backends:
        pred = load_predictor(Settings(_env_file=None, model_backend=name, db_enabled=False))  # type: ignore[call-arg]
        t0 = time.perf_counter()
        proba = pred.predict_proba(x)
        batch_ms = (time.perf_counter() - t0) * 1000
        diff = np.abs(proba - ref)
        flips = int(((proba >= threshold) != (ref >= threshold)).sum())
        res = {
            "backend": name,
            "n": len(x),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "decision_flips": flips,
            "tolerance": TOLERANCE.get(name, 1e-6),
            "passed": bool(diff.max() < TOLERANCE.get(name, 1e-6) and flips == 0),
            "batch_5000_ms": round(batch_ms, 1),
        }
        ok &= res["passed"]
        (args.out / f"regression_{name}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res))
    print("NON-RÉGRESSION :", "OK" if ok else "ÉCHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
