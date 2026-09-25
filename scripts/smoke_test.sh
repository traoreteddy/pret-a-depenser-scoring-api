#!/usr/bin/env bash
# Test de fumée de l'API : attend la disponibilité, vérifie /predict (200 et 422) et /metrics.
# Usage : scripts/smoke_test.sh [URL] [timeout_s]
set -euo pipefail

URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-90}"
FIXTURE="$(dirname "$0")/../tests/fixtures/client_ok.json"

echo "→ attente de ${URL}/health/ready (max ${TIMEOUT}s)"
for i in $(seq 1 "$TIMEOUT"); do
  if curl -fsS "${URL}/health/ready" >/dev/null 2>&1; then
    echo "  prêt après ${i}s"
    break
  fi
  if [ "$i" -eq "$TIMEOUT" ]; then
    echo "✗ API non disponible après ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 1
done

echo "→ GET /health/ready"
curl -fsS "${URL}/health/ready" | tee /dev/stderr | grep -q '"model_loaded":true'

echo "→ POST /predict (dossier valide)"
resp=$(curl -fsS -X POST "${URL}/predict" -H 'Content-Type: application/json' --data @"$FIXTURE")
echo "  $resp"
echo "$resp" | grep -Eq '"proba_defaut":(0(\.[0-9]+)?|1(\.0+)?)' || { echo "✗ proba_defaut absente ou hors [0,1]" >&2; exit 1; }
echo "$resp" | grep -Eq '"decision":"(Accordé|Refusé)"' || { echo "✗ décision invalide" >&2; exit 1; }

echo "→ POST /predict (âge négatif → 422 attendu)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${URL}/predict" -H 'Content-Type: application/json' \
  --data "$(sed 's/"DAYS_BIRTH": -14000/"DAYS_BIRTH": 1826/' "$FIXTURE")")
[ "$code" = "422" ] || { echo "✗ code ${code} au lieu de 422" >&2; exit 1; }
echo "  422 OK"

echo "→ GET /metrics"
curl -fsS "${URL}/metrics" | grep -q '^predictions_total' || { echo "✗ métrique predictions_total absente" >&2; exit 1; }
echo "  métriques OK"

echo "✓ smoke test réussi"
