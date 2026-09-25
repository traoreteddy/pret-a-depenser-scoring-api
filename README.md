# Prêt à Dépenser — API de scoring crédit : déploiement, monitoring et optimisation

[![CI/CD](https://github.com/traoreteddy/pret-a-depenser-scoring-api/actions/workflows/ci.yml/badge.svg)](https://github.com/traoreteddy/pret-a-depenser-scoring-api/actions/workflows/ci.yml)

Projet OpenClassrooms **« Déployez et monitorez votre modèle de scoring »** (MLOps 2/2).
Mise en production du modèle de scoring **XGBoost** versionné avec MLflow au projet précédent
(`scoring_credit` v1, alias `champion`, run `4cad9144…`, 38 variables, seuil métier 0,48) :

- **API FastAPI** (`POST /predict`) avec validation Pydantic stricte, sondes de vivacité et de
  disponibilité, gestion d'erreurs uniforme, Swagger ; modèle chargé **une seule fois** au démarrage ;
- **Tests automatisés** (pytest, couverture ≥ 85 %), lint (ruff), typage (mypy) ;
- **Docker** (image multi-stage, non-root, healthcheck) et **docker-compose** complet ;
- **CI/CD GitHub Actions** : lint → typage → tests → image publiée sur GHCR → déploiement simulé ;
- **Stockage des données de production** : PostgreSQL (prédictions, entrées échantillonnées, erreurs, labels différés) ;
- **Monitoring technique** : métriques Prometheus + dashboard Grafana provisionné ;
- **Analyse du data drift** : notebook (Evidently, PSI/KS, estimation de performance, calibrage du seuil) et dashboard Streamlit ;
- **Optimisation post-déploiement** : profilage cProfile, backend XGBoost natif / ONNX Runtime, benchmarks reproductibles (voir [`docs/optimisation.md`](docs/optimisation.md)).

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Utiliser l'API](#2-utiliser-lapi)
3. [Structure du dépôt](#3-structure-du-dépôt)
4. [Tests et qualité](#4-tests-et-qualité)
5. [Docker et docker-compose](#5-docker-et-docker-compose)
6. [Pipeline CI/CD](#6-pipeline-cicd)
7. [Données de production et stockage](#7-données-de-production-et-stockage)
8. [Monitoring : Grafana, Streamlit, notebook](#8-monitoring--grafana-streamlit-notebook)
9. [Simulation de trafic et de dérive](#9-simulation-de-trafic-et-de-dérive)
10. [Optimisation des performances](#10-optimisation-des-performances)
11. [Configuration et secrets](#11-configuration-et-secrets)
12. [Reprise du modèle P6 et reproductibilité](#12-reprise-du-modèle-p6-et-reproductibilité)
13. [Perspectives : ré-entraînement automatique](#13-perspectives--ré-entraînement-automatique)

## 1. Démarrage rapide

Prérequis : [uv](https://docs.astral.sh/uv/) ≥ 0.11, Python 3.13 (installé par uv), Docker + Compose v2, `make`.

```bash
git clone https://github.com/traoreteddy/pret-a-depenser-scoring-api.git && cd pret-a-depenser-scoring-api
cp .env.example .env            # puis remplacer les valeurs « changez-moi »
make install                    # uv sync --all-extras --all-groups
make ci                         # ruff + mypy + pytest (couverture)
make run                        # API locale : http://localhost:8000/docs
```

Stack complète (API + PostgreSQL + pgAdmin + Prometheus + Grafana) :

```bash
make up                         # docker compose --profile tools up -d --build
make smoke                      # test de fumée contre http://localhost:8000
make simulate SCENARIO=normal   # 2 000 dossiers réels à 50 req/s
make dashboard                  # Streamlit : http://localhost:8501
```

| Service | URL | Identifiants |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| Métriques Prometheus | http://localhost:8000/metrics | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / `GRAFANA_ADMIN_PASSWORD` |
| pgAdmin | http://localhost:5050 | `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` (serveur pré-enregistré) |
| PostgreSQL | localhost:5433 | `POSTGRES_USER` / `POSTGRES_PASSWORD` |
| Streamlit | http://localhost:8501 | — |

`make help` liste toutes les cibles.

## 2. Utiliser l'API

### `POST /predict`

Entrée : les **32 champs bruts** d'un dossier (voir `tests/fixtures/client_ok.json`). L'API calcule
elle-même les 9 variables dérivées du notebook P6 (`EXT_SOURCE_MEAN/MIN/STD/NB_NAN`,
`CREDIT_ANNUITY_RATIO`, `CREDIT_GOODS_RATIO`, `AGE_ANNEES`, `ID_PUBLISH_AGE_RATIO`,
`PREV_REFUSED_RATIO`), ce qui garantit leur cohérence et rend la validation métier naturelle.

```bash
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
     --data @tests/fixtures/client_ok.json | python -m json.tool
```

```json
{
  "request_id": "0a1f…", "proba_defaut": 0.4259, "decision": "Accordé", "threshold": 0.48,
  "model_name": "scoring_credit", "model_version": "1", "model_backend": "xgb_native", "latency_ms": 0.31
}
```

Règle métier (P6) : `decision = "Refusé"` si `proba_defaut ≥ threshold`, sinon `"Accordé"`.
En-têtes de réponse : `X-Request-ID` (repris de la requête s'il est fourni), `X-Process-Time-Ms`.
En-têtes optionnels de requête : `X-Client-Ref` (identifiant technique pour rattacher un label
différé), `X-Scenario` (marquage du trafic simulé ; force la journalisation complète).

### Validation et erreurs

Mode **strict** : types exacts (`"30000"` refusé pour un nombre, `true` refusé pour un flag 0/1),
champs inconnus refusés, `null` accepté uniquement pour les champs optionnels. Bornes de
plausibilité (couvrent 100 % des données d'entraînement) : `AMT_CREDIT > 0`, `DAYS_BIRTH` entre
−36 525 et −6 570 (18 à 100 ans), `EXT_SOURCE_*` dans [0, 1], `DAYS_*` ≤ 0, etc. Cohérences
inter-champs : `PREV_REFUSED_COUNT ≤ PREV_COUNT`, `BUREAU_DAYS_CREDIT_MIN ≤ MAX`.

| Code | `error` | Cas |
|---|---|---|
| 422 | `validation_error` | champ manquant, hors plage (âge −5 ans, crédit 0), mauvais type, champ inconnu, JSON malformé — `details[].loc/msg/type` |
| 500 | `model_error` / `internal_error` | échec d'inférence ; jamais de stack trace au client, erreur journalisée |
| 503 | `not_ready` | modèle non chargé |

> Le brief cite « un revenu de 0 » : `AMT_INCOME_TOTAL` n'est pas une variable du modèle retenu ;
> le cas équivalent est `AMT_CREDIT = 0` (ou `AMT_ANNUITY = 0`), rejeté par `gt=0`.

### Sondes

- `GET /health/live` : le processus répond (200).
- `GET /health/ready` : 200 si le modèle est chargé (503 sinon) ; indique l'état de la base
  (`ok` / `degraded` / `disabled`). `READY_REQUIRE_DB=true` rend la base obligatoire.

## 3. Structure du dépôt

```
├── src/scoring_api/            code source (package installable)
│   ├── main.py                 création de l'app, lifespan (chargement unique du modèle)
│   ├── config.py               Settings pydantic-settings (.env)
│   ├── api/                    routes, schémas Pydantic, erreurs, middleware
│   ├── features/               spec des 38 variables + port de FeatureEngineer
│   ├── model/                  Protocol Predictor, backends pyfunc / xgb_native / onnx, registre
│   ├── storage/                schema.sql, pool psycopg3, journal asynchrone, échantillonnage
│   ├── observability/          logs JSON structlog, métriques Prometheus
│   └── monitoring/             dérive (PSI/KS/Evidently), performance (CBPE), seuil, accès données
├── models/scoring_credit_v1/   artefacts MLflow du champion P6 + booster.ubj + model.onnx + model_meta.json
├── data/                       référence (20 k), production simulée (5 k), labels, métriques de référence
├── scripts/                    import/export du modèle, données, non-régression, profilage, labels, smoke, retrain
├── simulation/                 simulateur de trafic et scénarios de dérive
├── benchmarks/                 bench.py, run_matrix.sh, results/
├── notebooks/drift_analysis.ipynb
├── dashboard/app.py            Streamlit
├── monitoring/                 prometheus.yml, provisioning Grafana + dashboard JSON, pgadmin
├── docs/                       architecture, monitoring, optimisation, captures d'écran
├── tests/                      unit/ api/ integration/ fixtures/
├── Dockerfile · Dockerfile.dashboard · docker-compose.yml · Makefile · pyproject.toml · uv.lock
└── .github/workflows/ci.yml
```

## 4. Tests et qualité

```bash
make lint        # ruff check + format --check
make typecheck   # mypy --strict sur src/
make test        # tests rapides (prédicteur factice), couverture ≥ 85 %
make test-all    # + modèle réel (slow) + PostgreSQL (integration, DATABASE_URL requis)
```

| Niveau | Fichiers | Ce qui est vérifié |
|---|---|---|
| Unitaire | `tests/unit/test_features.py` | les 9 formules dérivées à la main et contre `input_example.json` (1e-9), std ddof=1, NaN, ordre des 38 colonnes = signature MLflow |
| Unitaire | `test_schemas.py`, `test_sampling.py`, `test_monitoring.py`, `test_model_backends.py` | contraintes Pydantic, échantillonnage déterministe, PSI/KS/coût, égalité des backends avec les probabilités P6 |
| API | `tests/api/` | nominal, `X-Request-ID`, champs manquants, hors plage (âge −5, crédit 0), types invalides (`"abc"`, `"30000"`, booléen), champ inconnu, JSON malformé, 500 sans fuite, 503 avant chargement, `/metrics`, `/docs` |
| Modèle réel | `test_real_model.py` (`slow`) | l'API redonne exactement les probabilités du pyfunc MLflow |
| Intégration | `tests/integration/` | 300 requêtes → 300 lignes en base, échantillonnage ≈ 25 %, en-têtes, erreurs 422 persistées |

Couverture actuelle : ~95 % (`make cov` → `htmlcov/index.html`).

## 5. Docker et docker-compose

```bash
make docker-build                        # image scoring-api:local (multi-stage, non-root, HEALTHCHECK)
docker run --rm -p 8000:8000 -e DB_ENABLED=false scoring-api:local
```

`INSTALL_EXTRAS` (build-arg) choisit les dépendances : `""` = image allégée pour le backend natif
(par défaut), `pyfunc` = baseline MLflow, `pyfunc onnx` = tous les backends (benchmarks).

`docker-compose.yml` : `api`, `db` (PostgreSQL 17, port hôte 5433), `prometheus`, `grafana`,
`pgadmin` (profil `tools`), `dashboard` (profil `dashboard`). Ports hôte modifiables dans `.env`.

```bash
make up            # api + db + prometheus + grafana + pgadmin
make logs          # logs JSON de l'API
make down          # arrêt + suppression des volumes
docker compose --profile dashboard up -d dashboard   # Streamlit conteneurisé
```

## 6. Pipeline CI/CD

`.github/workflows/ci.yml`, déclenché sur `push`/`pull_request` vers `main` et sur les tags `v*` :

| Job | Contenu |
|---|---|
| `lint` | ruff check + format |
| `typecheck` | mypy |
| `test` | pytest avec service PostgreSQL 17 (tests d'intégration), modèle réel, `--cov-fail-under=85`, artefact `coverage.xml` |
| `build-push` | buildx, cache GHA, tags `sha-<short>` + `latest`, push sur `ghcr.io/traoreteddy/pret-a-depenser-scoring-api` (hors PR) |
| `deploy-simulated` | `.env` éphémère depuis les secrets, `docker compose up` sur l'image publiée, `scripts/smoke_test.sh` (ready, 200, 422, /metrics), logs si échec, `down -v` |

Secrets GitHub : `POSTGRES_PASSWORD`, `GRAFANA_ADMIN_PASSWORD` (valeurs par défaut de CI sinon) ;
`GITHUB_TOKEN` automatique pour GHCR. Variables : `MODEL_BACKEND`, `DEPLOY_DB_ENABLED`.
Démonstration : un commit sur `main` → tests → image → déploiement, visible dans l'onglet Actions.

## 7. Données de production et stockage

Chaque appel à `/predict` est journalisé dans PostgreSQL par un **journal asynchrone**
(file bornée + insertion par lots, quelques µs côté requête ; la base ne peut pas ralentir l'API).

| Table | Contenu |
|---|---|
| `predictions` | 1 ligne par prédiction : `request_id`, `ts`, modèle/version/backend, seuil, `proba_defaut`, `decision`, latences (totale, features, inférence), `status_code`, `sampled`, `raw_input` et `features` (JSONB, si échantillonné), `client_ref`, `scenario` |
| `api_errors` | erreurs 422 : détail Pydantic (`loc`, `msg`, `type`) et corps reçu |
| `prediction_labels` | labels différés (`target`) rattachés par `request_id` |

**Échantillonnage** (`LOG_SAMPLE_RATE=0.25`, `LOG_GREY_ZONE=0.05`) : 100 % des scores et
latences (≈ 120 o/ligne), entrées complètes (≈ 2 Ko) pour 25 % des requêtes (tirage déterministe
sur `request_id`), **toujours** en zone grise `|proba − seuil| < 0,05` et pour le trafic marqué
`X-Scenario`. Ordre de grandeur : 1 M requêtes/mois ≈ 120 Mo de lignes légères + ≈ 500 Mo
d'entrées. Pas de donnée personnelle directe (RGPD) ; purge/partitionnement mensuel par `ts`
en perspective.

Captures d'écran de la solution de stockage et du monitoring (`docs/screenshots/`) :

| Capture | Contenu |
|---|---|
| `pgadmin_predictions.png` | table `predictions` : scores, décisions, latences, entrées JSONB (`raw_input`, `features`), `client_ref`, `scenario` |
| `pgadmin_par_scenario.png` | agrégats par scénario (volume, entrées complètes, score moyen, taux de refus, p99, labels) |
| `grafana_dashboard.png` | dashboard technique (RPS, p50/p95/p99 vs SLO, refus, heatmap des scores, journalisation) |
| `streamlit_*.png` | pages Métier, Dérive (dont scénario `drift_ext_source`), Performance, Seuil, Technique |

Requête utile :

```sql
SELECT scenario, count(*), round(avg(proba_defaut)::numeric, 3) AS score_moyen,
       round(avg((decision = 'Refusé')::int)::numeric, 3) AS taux_refus,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_total_ms) AS p99_ms
FROM predictions GROUP BY 1 ORDER BY 1;
```

## 8. Monitoring : Grafana, Streamlit, notebook

Guide d'interprétation détaillé : [`docs/monitoring.md`](docs/monitoring.md).

- **Grafana** (technique, temps réel) : RPS, p50/p95/p99 avec ligne SLO 180 ms, taux d'erreur,
  taux de refus vs 32 %, heatmap des scores, inférence par backend, file de journalisation,
  part des requêtes < 180 ms. Métriques : `http_request_duration_seconds` (bucket 0,18 s),
  `model_inference_duration_seconds{backend}`, `predictions_total{decision}`,
  `prediction_errors_total`, `prediction_score`, `prediction_log_*`, `model_info`, `app_ready`.
- **Streamlit** (métier + dérive) : pages Métier, Dérive des données, Performance, Seuil de
  décision, Technique ; filtres fenêtre / scénario.
- **Notebook** `notebooks/drift_analysis.ipynb` : analyse automatique reproductible, scénario par
  scénario (heatmap PSI, Evidently, IC du taux de refus, calibration, performance attendue vs
  réalisée, courbe de coût, synthèse et points de vigilance). `make drift-report` produit la même
  analyse en ligne de commande (CSV + rapport Evidently HTML dans `docs/reports/`).

Seuils de décision : PSI < 0,10 stable · 0,10–0,25 à surveiller · ≥ 0,25 dérive ; KS p < 0,01 /
n colonnes ; dérive « dataset » si > 30 % des colonnes en alerte (≥ 200 lignes).

## 9. Simulation de trafic et de dérive

Les dossiers envoyés sont **réels** (X_test du P6, disjoints de la référence) ; les scénarios
appliquent des transformations déterministes (graine fixe) :

```bash
make simulate SCENARIO=normal            # distribution d'entraînement
make simulate SCENARIO=drift_ext_source  # scores externes −0,15 (population plus risquée)
make simulate SCENARIO=drift_credit      # montants ×1,5
make simulate SCENARIO=drift_young       # uniquement < 30 ans
make simulate SCENARIO=drift_missing     # EXT_SOURCE_1/3 absents (panne fournisseur)
make simulate SCENARIO=invalid           # 5 % de dossiers invalides → 422
make labels                              # rattache les labels différés (TARGET réel)
```

Chaque simulation écrit un résumé (codes HTTP, p50/p95/p99 client et serveur) dans
`simulation/results/`.

## 10. Optimisation des performances

Rapport complet : [`docs/optimisation.md`](docs/optimisation.md) (profilage, matrice de
benchmarks, non-régression, justification de la configuration finale).

**Résumé** (API en Docker, 2 CPU, dossiers réels, client de charge multi-processus, médiane de 2 répétitions) :

| Configuration | p99 c=1 | p99 c=8 | p99 c=32 | p99 c=64 | Débit max |
|---|---|---|---|---|---|
| Avant : pyfunc MLflow (DataFrame + sklearn), pool de threads | 10,8 ms | 99 ms | 315 ms | 619 ms | 145 req/s |
| **Après : booster XGBoost natif, `nthread=1`, exécution inline** | **3,9 ms** | **8 ms** | **77 ms** | **75 ms** | **1 560 req/s** |

- Profilage : 80 % du temps du pyfunc était dans la construction du DataFrame et l'enrobage
  scikit-learn, pas dans XGBoost (1,2 ms) → appel direct du `Booster` (`inplace_predict`, float32).
- Non-régression : écart de probabilité **nul** sur 5 000 dossiers, 0 décision changée.
- Image Docker : 1,34 Go → **465 Mo** (sans MLflow/sklearn/pandas, roue `xgboost-cpu`), chargement du modèle 2 s → 0,2 s.
- ONNX Runtime, `nthread=-1` et 2 workers ont été mesurés et écartés (pas de gain à 2 CPU, voir le rapport).

## 11. Configuration et secrets

Toute la configuration passe par des variables d'environnement (`src/scoring_api/config.py`,
pydantic-settings) ; `.env.example` les documente, `.env` n'est **jamais commité**. En CI, les
secrets GitHub alimentent un `.env` éphémère. Principales variables :

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_BACKEND` | `xgb_native` | `pyfunc` (MLflow), `xgb_native` (Booster), `onnx` |
| `MODEL_THRESHOLD` | seuil du modèle (0,48) | surcharge du seuil métier |
| `XGB_NTHREAD` | 1 | threads du booster (1 = pas de contention entre requêtes) |
| `DB_ENABLED` / `DATABASE_URL` | true / — | journalisation PostgreSQL |
| `LOG_SAMPLE_RATE`, `LOG_GREY_ZONE` | 0,25 / 0,05 | échantillonnage des entrées |
| `LOG_QUEUE_MAXSIZE`, `LOG_BATCH_SIZE`, `LOG_FLUSH_INTERVAL_S` | 10 000 / 200 / 0,5 | file de journalisation |
| `READY_REQUIRE_DB` | false | la sonde ready exige la base |
| `METRICS_ENABLED` | true | endpoint `/metrics` |
| `WEB_CONCURRENCY` | 1 | workers uvicorn |

## 12. Reprise du modèle P6 et reproductibilité

- `scripts/import_model.py` copie les artefacts MLflow du champion (`MLmodel`, `python_model.pkl`,
  `input_example.json`, …) et écrit `model_meta.json` (run_id, seuil, sha256, 38 variables).
- `scripts/export_model.py` extrait le booster XGBoost (`booster.ubj`), le convertit en ONNX et
  enregistre la **permutation de colonnes** imposée par le `ColumnTransformer` du pipeline.
- `scripts/build_reference_data.py` reproduit exactement le découpage du notebook P6
  (`test_size=0.2, stratify, random_state=42`) et retrouve les métriques du P6 : AUC 0,778,
  coût 0,501, taux de refus 32,1 %, défaut 8,07 %.
- Versions épinglées (`uv.lock`) pour dépickler : Python 3.13, scikit-learn 1.9.1, xgboost 3.4.1,
  mlflow 3.16.1, pandas 3.0.5, numpy 2.5.3. Le backend natif ne dépend plus du pickle.

## 13. Perspectives : ré-entraînement automatique

`scripts/retrain_stub.py` formalise les déclencheurs (dérive du jeu de données, coût réalisé
> référence + 10 % avec assez de labels, calendrier) et les étapes (extraction des dossiers
labellisés, ré-entraînement du pipeline P6, comparaison au champion sur validation figée,
enregistrement MLflow en `challenger`, promotion manuelle, redéploiement par la CI). Autres
pistes : partitionnement de `predictions`, alerting Grafana/Alertmanager sur le p99 et le taux de
refus, OpenTelemetry (`OTEL_ENABLED`) pour le traçage distribué.
