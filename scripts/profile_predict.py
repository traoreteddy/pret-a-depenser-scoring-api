"""Profilage cProfile du chemin /predict (en processus, sans réseau) pour un backend donné.

Usage : uv run python scripts/profile_predict.py [--backend pyfunc] [--n 500]
Sortie : profiles/predict_<backend>.prof (+ top des fonctions par temps cumulé sur stdout).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi.testclient import TestClient

from scoring_api.api.schemas import EXAMPLE_CLIENT
from scoring_api.config import Settings
from scoring_api.main import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="pyfunc", choices=["pyfunc", "xgb_native", "onnx"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    settings = Settings(
        _env_file=None,
        model_backend=args.backend,
        db_enabled=False,
        log_level="ERROR",
        log_format="console",
    )  # type: ignore[call-arg]
    app = create_app(settings)
    out = Path("profiles")
    out.mkdir(exist_ok=True)
    with TestClient(app) as client:
        for _ in range(50):  # échauffement
            client.post("/predict", json=EXAMPLE_CLIENT)
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        for _ in range(args.n):
            client.post("/predict", json=EXAMPLE_CLIENT)
        prof.disable()
        elapsed = time.perf_counter() - t0

        # Mesure directe de l'inférence seule (hors HTTP/validation)
        import numpy as np  # noqa: PLC0415

        from scoring_api.features.engineering import build_features  # noqa: PLC0415

        x = build_features(EXAMPLE_CLIENT)
        pred = client.app.state.predictor  # type: ignore[attr-defined]
        ts = []
        for _ in range(args.n):
            t = time.perf_counter()
            pred.predict_proba(x)
            ts.append((time.perf_counter() - t) * 1000)

    path = out / f"predict_{args.backend}.prof"
    prof.dump_stats(str(path))
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
    stats.print_stats(args.top)
    print(
        f"backend={args.backend}  {args.n} requêtes en {elapsed:.2f} s → {elapsed / args.n * 1000:.2f} ms/requête (TestClient, en processus)"
    )
    print(
        f"inférence seule : p50 {np.percentile(ts, 50):.3f} ms, p99 {np.percentile(ts, 99):.3f} ms"
    )
    print(stream.getvalue())
    print(f"profil : {path}  (visualiser : uv run snakeviz {path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
