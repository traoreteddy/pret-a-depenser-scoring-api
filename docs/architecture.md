# Architecture

```
                      ┌──────────────────────────────────────────────────────────────┐
   client / simulateur│  FastAPI  (src/scoring_api)                                  │
  POST /predict ─────▶│  validation Pydantic (32 champs bruts, strict)               │
  X-Client-Ref        │  features/engineering.py  → 38 variables (ordre signature)   │
  X-Scenario          │  model/registry.py → Predictor chargé UNE fois (lifespan)     │
                      │     pyfunc (MLflow) | xgb_native (Booster) | onnx (ORT)       │
                      │  storage/logger.py → asyncio.Queue → lots → PostgreSQL        │
                      │  observability/metrics.py → /metrics (Prometheus)            │
                      └───────────┬───────────────────────────┬──────────────────────┘
                                  │ INSERT par lots            │ scrape 5 s
                       ┌──────────▼──────────┐      ┌─────────▼─────────┐
                       │ PostgreSQL          │      │ Prometheus         │──▶ Grafana
                       │ predictions         │      │ (rétention 7 j)    │   dashboard
                       │ api_errors          │      └───────────────────┘   technique
                       │ prediction_labels   │
                       └──────────┬──────────┘
                                  │ lecture (pandas)
                 ┌────────────────┴────────────────┐
                 │ notebooks/drift_analysis.ipynb  │   Evidently, PSI/KS, CBPE, seuil
                 │ dashboard/app.py (Streamlit)    │   métier + dérive + performance
                 └─────────────────────────────────┘
```

## Composants

| Composant | Rôle | Fichiers |
|---|---|---|
| API | Exposition du modèle, validation, erreurs uniformes, sondes | `src/scoring_api/api/`, `main.py` |
| Features | Port exact de `FeatureEngineer` (notebook P6) | `src/scoring_api/features/` |
| Modèle | Backends interchangeables derrière le `Protocol Predictor` | `src/scoring_api/model/` |
| Stockage | Schéma SQL, pool psycopg3, journal asynchrone, échantillonnage | `src/scoring_api/storage/` |
| Observabilité | Logs JSON (structlog), métriques Prometheus | `src/scoring_api/observability/` |
| Monitoring | Dérive, performance, seuil (partagé notebook / Streamlit / CLI) | `src/scoring_api/monitoring/` |
| Simulation | Trafic réaliste et scénarios de dérive | `simulation/` |
| Benchmarks | Latence HTTP reproductible, non-régression | `benchmarks/`, `scripts/check_regression.py` |
| Infra | Dockerfile multi-stage, compose (api, db, pgadmin, prometheus, grafana, dashboard), CI | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Flux d'une requête `/predict`

1. Middleware ASGI : `request_id` (généré ou repris de `X-Request-ID`), chronométrage, en-têtes
   `X-Request-ID` / `X-Process-Time-Ms`, log d'accès JSON, métriques HTTP.
2. Validation Pydantic stricte → 422 avec détail (`loc`, `msg`, `type`) ; l'erreur est journalisée
   dans `api_errors` avec le corps reçu.
3. `build_features` : 32 bruts → vecteur (1, 38) float64 (NaN pour les manquants).
4. `predictor.predict_proba` (backend choisi par `MODEL_BACKEND`) → probabilité de défaut.
5. Décision : `Refusé` si proba ≥ seuil (0,48 par défaut, `MODEL_THRESHOLD` pour surcharger).
6. Métriques (latence features / inférence, score, décision) et mise en file du journal
   (non bloquant, thread-safe via `call_soon_threadsafe`).
7. Réponse `PredictionResponse`.

## Choix techniques

- **FastAPI + Pydantic v2 strict** : validation exhaustive, documentation OpenAPI automatique.
- **Endpoint synchrone** : exécuté dans le pool de threads de Starlette, l'inférence CPU ne bloque
  pas la boucle d'événements ; la journalisation est découplée par une file asynchrone.
- **Modèle embarqué dans l'image** : pas de dépendance au serveur MLflow en production ; le
  `run_id` et le `sha256` du pickle sont dans `model_meta.json` pour la traçabilité.
- **Backend natif par défaut** (`booster.ubj`) : plus de pickle ni de scikit-learn/pandas au
  runtime, image allégée, latence divisée (voir `docs/optimisation.md`). Le backend `pyfunc`
  reste disponible comme référence de non-régression.
- **PostgreSQL** : requêtable (SQL, pandas), JSONB pour les entrées, index partiels, labels
  différés joints par `request_id`. Échantillonnage pour maîtriser le volume.
- **Prometheus/Grafana** pour le temps réel technique ; **Streamlit** pour l'analyse métier et la
  dérive (lecture PostgreSQL, calculs partagés avec le notebook).
