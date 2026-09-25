#!/usr/bin/env bash
# Matrice de benchmarks : relance l'API (docker compose, limite 2 CPU / 2 Go) avec chaque
# configuration et exécute benchmarks/bench.py (client multi-processus, 4 processus).
# Usage : bash benchmarks/run_matrix.sh [n] [repeat]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-2000}"; REPEAT="${2:-2}"
run() {  # tag backend nthread workers threadpool note
  local tag=$1 backend=$2 nthread=$3 workers=$4 tp=$5 note=$6
  echo "=== $tag ($note) ==="
  MODEL_BACKEND=$backend XGB_NTHREAD=$nthread WEB_CONCURRENCY=$workers PREDICT_IN_THREADPOOL=$tp \
    docker compose up -d --no-build --force-recreate api >/dev/null 2>&1
  for i in $(seq 1 60); do curl -fsS localhost:8000/health/ready >/dev/null 2>&1 && break; sleep 1; done
  uv run python benchmarks/bench.py --tag "$tag" --n "$N" --repeat "$REPEAT" --concurrency 1 8 32 64 --procs 4 --note "$note"
}
run 1_pyfunc_baseline        pyfunc     -1 1 true  "AVANT : pyfunc MLflow (DataFrame + sklearn, XGB n_jobs=-1), endpoint dans le pool de threads, 1 worker"
run 2_pyfunc_inline          pyfunc     -1 1 false "pyfunc MLflow, exécution inline"
run 3_xgb_native_threadpool  xgb_native 1  1 true  "Booster natif (inplace_predict, nthread=1), pool de threads"
run 4_xgb_native_inline      xgb_native 1  1 false "APRÈS : Booster natif, nthread=1, exécution inline, 1 worker"
run 5_xgb_native_inline_tall xgb_native -1 1 false "Booster natif, nthread=-1 (tous les cœurs), inline"
run 6_onnx_inline            onnx       1  1 false "ONNX Runtime CPU (intra_op=1), inline"
run 7_xgb_native_inline_w2   xgb_native 1  2 false "Booster natif, inline, 2 workers uvicorn"
MODEL_BACKEND=xgb_native XGB_NTHREAD=1 WEB_CONCURRENCY=1 PREDICT_IN_THREADPOOL=false docker compose up -d --no-build --force-recreate api >/dev/null 2>&1
echo "matrice terminée"
