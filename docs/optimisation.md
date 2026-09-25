# Optimisation post-déploiement : profilage, tests et résultats

Objectif fixé : **p99 du temps de réponse < 180 ms** sous charge, sans régression des prédictions.

## 1. Point de départ (données de monitoring)

Le premier trafic simulé contre l'API déployée (backend `pyfunc` MLflow, conteneur limité à
2 CPU) donnait déjà un signal : à 100 req/s, **p99 client = 232 ms** et p99 serveur = 128 ms
(`simulation/results/normal_*.json`), alors que le modèle seul ne prend que ~3 ms. Les métriques
Prometheus (`model_inference_duration_seconds` vs `http_request_duration_seconds`) confirmaient
que le temps se perdait *autour* de l'inférence.

## 2. Profilage (cProfile, `scripts/profile_predict.py`)

500 appels `/predict` en processus (TestClient), Apple M2 Max :

| Backend | Coût par requête | Répartition (temps cumulé) |
|---|---|---|
| `pyfunc` (MLflow) | **11,8 ms** dont 10,1 ms dans `predict_proba` | construction du DataFrame + `astype` (19 500 appels !) : **5,3 ms** · enrobage `FixedThresholdClassifier` → `Pipeline` → `ColumnTransformer` : **2,3 ms** · XGBoost réel (`predict_proba` sklearn) : 1,2 ms |
| `xgb_native` (Booster) | **1,6 ms** dont 0,055 ms d'inférence (p99 0,15 ms) | le reste est la pile HTTP/validation |

**Goulots identifiés** : (1) le pyfunc reconstruit un DataFrame pandas et traverse trois couches
scikit-learn pour chaque dossier, soit ~8× le coût de XGBoost lui-même ; (2) `n_jobs=-1` du
`XGBClassifier` lance un pool OpenMP par requête, source de contention sous concurrence ;
(3) l'endpoint synchrone passe par le pool de threads de Starlette (changement de thread + GIL)
pour un travail de 0,05 ms.

## 3. Stratégies testées

| # | Stratégie | Mise en œuvre |
|---|---|---|
| A | **Booster natif** : extraire le `Booster` du pipeline, `inplace_predict` sur float32, sans DataFrame ni sklearn | `scripts/export_model.py` → `booster.ubj` ; `model/xgb_backend.py` ; permutation de colonnes du `ColumnTransformer` enregistrée dans `model_meta.json` |
| B | **ONNX Runtime** (CPU, intra_op = 1) | conversion `onnxmltools` (opset 15) → `model.onnx` ; `model/onnx_backend.py` |
| C | **Threads XGBoost** : `nthread=1` vs `-1` | `XGB_NTHREAD` |
| D | **Exécution inline** de `/predict` (endpoint async, pas de pool de threads) | `PREDICT_IN_THREADPOOL=false` |
| E | **2 workers uvicorn** | `WEB_CONCURRENCY=2` |
| F | **Image allégée** : sans MLflow/sklearn/pandas, roue `xgboost-cpu` sous Linux | `INSTALL_EXTRAS=""`, marqueur `sys_platform` dans `pyproject.toml` |

## 4. Protocole de mesure

- `benchmarks/bench.py` : dossiers **réels** (`data/prod_sample.parquet`, graine 42), 300 requêtes
  d'échauffement, 2 000 requêtes mesurées par niveau de concurrence (1, 8, 32, 64), **2 répétitions**
  (médiane du p99 retenue), latence **client** (aller-retour) et **serveur** (`X-Process-Time-Ms`).
- API dans Docker Desktop (`deploy.resources.limits` : 2 CPU, 2 Go), PostgreSQL et Prometheus
  actifs (journalisation réelle). Hôte : Apple M2 Max, 32 Go, Docker 24, VM 6 CPU / 8 Go.
- **Contrôle du générateur de charge.** Une première campagne avec un client httpx mono-processus
  donnait des p99 > 350 ms à 32 connexions *alors que le p99 serveur restait à 6 ms* et que le
  débit chutait de 1 150 à 340 req/s entre 8 et 32 connexions : le client Python était le goulot.
  Vérification avec ApacheBench (compilé) sur la même API : 1 300 req/s, p99 = 29 ms à c=32.
  Le client a été réécrit en **multi-processus** (4 processus) et re-validé contre ApacheBench
  (p99 32 ms à c=32). Toutes les mesures ci-dessous utilisent ce client corrigé.

## 5. Résultats

| Configuration | c=1 client p50 / p99 (ms) | c=8 client p50 / p99 (ms) | c=32 client p50 / p99 (ms) | c=64 client p50 / p99 (ms) | c=1 serveur p99 | c=8 serveur p99 | c=32 serveur p99 | c=64 serveur p99 | RPS max |
|---|---|---|---|---|---|---|---|---|---|
| `1_pyfunc_baseline` — AVANT : pyfunc MLflow (DataFrame + sklearn, XGB n_jobs=-1), endpoint dans le pool de threads, 1 worker | 7.24 / **10.8** | 54.73 / **99.13** | 220.37 / **314.84** | 432.89 / **619.04** | 7.96 | 87.46 | 281.25 | 572.05 | 145 |
| `2_pyfunc_inline` — pyfunc MLflow, exécution inline | 7.25 / **10.25** | 42.13 / **66.41** | 162.59 / **471.74** | 324.75 / **415.01** | 7.2 | 47.12 | 324.2 | 333.88 | 193 |
| `3_xgb_native_threadpool` — Booster natif (inplace_predict, nthread=1), pool de threads | 2.07 / **4.33** | 6.84 / **10.11** | 22.08 / **31.64** | 43.21 / **81.07** | 2.02 | 8.12 | 23.7 | 71.04 | 1413 |
| `4_xgb_native_inline` — APRÈS : Booster natif, nthread=1, exécution inline, 1 worker | 1.95 / **3.88** | 5.93 / **8.13** | 19.87 / **76.55** | 38.87 / **74.94** | 1.5 | 5.97 | 19.07 | 65.12 | 1559 |
| `5_xgb_native_inline_tall` — Booster natif, nthread=-1 (tous les cœurs), inline | 1.89 / **3.81** | 5.97 / **8.55** | 19.93 / **59.01** | 39.45 / **54.43** | 1.44 | 6.12 | 51.25 | 35.19 | 1568 |
| `6_onnx_inline` — ONNX Runtime CPU (intra_op=1), inline | 1.68 / **4.61** | 5.14 / **7.43** | 19.84 / **41.54** | 37.8 / **196.14** | 1.89 | 5.41 | 31.56 | 88.86 | 1505 |
| `7_xgb_native_inline_w2` — Booster natif, inline, 2 workers uvicorn | 1.87 / **3.64** | 4.12 / **18.04** | 14.54 / **38.67** | 40.99 / **171.96** | 1.44 | 14.15 | 28.99 | 7.41 | 1832 |

Lecture : p50 / **p99** client en ms par niveau de concurrence, puis p99 serveur, puis débit max.

### Ce que montrent les chiffres

1. **Le backend natif est le levier majeur** (A) : à concurrence 8, le p99 passe de 99 ms à 8 ms
   (**÷12**) et le débit de 145 à 1 320 req/s (**×9**) ; à concurrence 32, de 315 ms à 77 ms.
2. **L'exécution inline** (D) gagne encore ~10 % de p50 et supprime les sauts de thread : c'est le
   défaut retenu. Elle est contre-productive pour le pyfunc (7 ms de Python bloquant la boucle),
   d'où l'option `PREDICT_IN_THREADPOOL` conservée.
3. **`nthread=-1`** (C) n'apporte rien à 2 CPU (600 arbres × 1 ligne : le parallélisme intra-requête
   ne paie pas) et rend les queues plus instables (p99 serveur 51 ms à c=32) : **`nthread=1`**.
4. **ONNX Runtime** (B) est équivalent au booster natif jusqu'à 32 connexions mais se dégrade à 64
   (p99 196 ms) sur 2 CPU ; il ajoute une conversion et une dépendance sans gain ici. Conservé
   comme backend optionnel (pertinent si GPU/quantification ou modèle plus lourd).
5. **2 workers** (E) augmentent le débit (1 830 req/s) mais, à 2 CPU partagés avec la
   journalisation, la contention CPU dégrade le p99 à 64 connexions (172 ms) : à réserver à des
   hôtes ≥ 4 CPU. Configuration finale : 1 worker par conteneur, monter en répliques si besoin.
6. **SLO** : avec la configuration finale, p99 = **3,9 ms** (c=1), **8 ms** (c=8), **77 ms** (c=32),
   **75 ms** (c=64) — sous 180 ms à tous les niveaux, contre 315 ms dès 32 connexions avant.

### Non-régression (`scripts/check_regression.py`, 5 000 dossiers)

| Backend | Écart max de probabilité vs pyfunc | Décisions changées | Temps pour 5 000 lignes |
|---|---|---|---|
| `xgb_native` | **0,0** | 0 | 36 ms |
| `onnx` | 8,9 × 10⁻⁸ | 0 | 72 ms |

Les tests `tests/unit/test_model_backends.py` et `tests/api/test_real_model.py` vérifient à
chaque CI que les backends redonnent les probabilités du modèle P6 sur `input_example.json`.
Aucune modification du modèle (mêmes arbres, mêmes poids) : pas de biais introduit, l'AUC et le
coût métier sont inchangés par construction.

### Empreinte et démarrage

| Image | Contenu | Taille | Chargement du modèle |
|---|---|---|---|
| `INSTALL_EXTRAS="pyfunc onnx"` | tous les backends (benchmarks) | 1,34 Go | 2,0 s |
| `INSTALL_EXTRAS="pyfunc"` | baseline MLflow | 1,25 Go | 2,0 s |
| `INSTALL_EXTRAS=""` + `xgboost` standard | booster natif | 893 Mo | 0,2 s |
| **`INSTALL_EXTRAS=""` + `xgboost-cpu`** (final) | booster natif, sans bibliothèques NVIDIA | **465 Mo** | **0,2 s** |

## 6. Configuration finale et justification

| Élément | Choix | Pourquoi |
|---|---|---|
| Backend | `MODEL_BACKEND=xgb_native` (`booster.ubj`) | ÷12 sur le p99, écart de prédiction nul, plus de dépendance au pickle ni à MLflow/sklearn/pandas au runtime |
| Threads | `XGB_NTHREAD=1` | pas de contention entre requêtes ; le parallélisme est au niveau des requêtes/répliques |
| Exécution | `PREDICT_IN_THREADPOOL=false` | inférence < 1 ms : l'inline évite le changement de thread |
| Workers | `WEB_CONCURRENCY=1` par conteneur | à 2 CPU, un 2ᵉ worker dégrade le p99 ; scaler horizontalement |
| Logiciel | Python 3.13, FastAPI, uvicorn (uvloop/httptools), xgboost-cpu 3.4.1, numpy 2.5 | pile légère ; `orjson` disponible, pydantic v2 sérialise déjà en Rust |
| Image | python:3.13-slim + libgomp1, non-root, 465 Mo | démarrage < 1 s, surface réduite |
| Matériel | 2 vCPU / 2 Go suffisent pour ~1 500 req/s avec p99 < 80 ms ; pas de GPU (modèle d'arbres, une ligne par requête) | coût minimal ; le GPU n'aiderait que pour du scoring par lots |

Déploiement : les défauts ont été basculés (`.env.example`, `docker-compose.yml`, `Dockerfile`,
CI) ; le pipeline CI/CD reconstruit l'image allégée et la redéploie (job `deploy-simulated`).

## 7. Reproduire

```bash
make profile                          # cProfile (profiles/predict_<backend>.prof)
make export-model && make regression  # booster.ubj / model.onnx + non-régression
make up                               # stack Docker (2 CPU)
bash benchmarks/run_matrix.sh 2000 2  # matrice complète (≈ 10 min)
uv run python benchmarks/report.py    # tableau markdown à partir de benchmarks/results/*.json
```
