#!/usr/bin/env bash
# Matrice de benchmarks : relance l'API (docker compose, 2 CPU) avec chaque configuration
# et exécute benchmarks/bench.py. Résultats dans benchmarks/results/<tag>.json.
# Usage : bash benchmarks/run_matrix.sh [n] [repeat]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-2000}"; REPEAT="${2:-2}"
run() {  # tag backend nthread workers note
  local tag=$1 backend=$2 nthread=$3 workers=$4 note=$5
  echo "=== $tag ($note) ==="
  MODEL_BACKEND=$backend XGB_NTHREAD=$nthread WEB_CONCURRENCY=$workers docker compose up -d --no-build --force-recreate api >/dev/null 2>&1
  for i in $(seq 1 60); do curl -fsS localhost:8000/health/ready >/dev/null 2>&1 && break; sleep 1; done
  uv run python benchmarks/bench.py --tag "$tag" --n "$N" --repeat "$REPEAT" --concurrency 1 8 32 --note "$note"
}
run pyfunc_baseline      pyfunc     1  1 "baseline : mlflow pyfunc (XGBClassifier n_jobs=-1), 1 worker"
run xgb_native_t1        xgb_native 1  1 "Booster natif inplace_predict, nthread=1, 1 worker"
run xgb_native_tall      xgb_native -1 1 "Booster natif, nthread=-1 (tous les cœurs), 1 worker"
run onnx_t1              onnx       1  1 "ONNX Runtime CPU, intra_op=1, 1 worker"
run xgb_native_t1_w2     xgb_native 1  2 "Booster natif, nthread=1, 2 workers uvicorn"
MODEL_BACKEND=pyfunc XGB_NTHREAD=1 WEB_CONCURRENCY=1 docker compose up -d --no-build --force-recreate api >/dev/null 2>&1
echo "matrice terminée"
