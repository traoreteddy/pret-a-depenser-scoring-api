# Makefile — Prêt à Dépenser scoring API
# Toutes les commandes Python passent par uv (environnement reproductible).

SHELL := /bin/bash
.DEFAULT_GOAL := help

-include .env
export

SCENARIO ?= normal
TAG      ?= baseline
N        ?= 2000
RPS      ?= 50
API_URL  ?= http://localhost:8000
COMPOSE  ?= docker compose

.PHONY: help install lock lint format typecheck test test-all cov run \
        import-model export-model data docker-build up down logs smoke \
        simulate labels bench profile regression drift-report dashboard notebook ci clean

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------- Environnement ----------
install: ## Installe toutes les dépendances (extras + groupes)
	uv sync --all-extras --all-groups

lock: ## Régénère uv.lock
	uv lock

# ---------- Qualité ----------
lint: ## Ruff (lint + format check)
	uv run ruff check .
	uv run ruff format --check .

format: ## Formate et corrige automatiquement
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Mypy sur src/
	uv run mypy

test: ## Tests rapides (sans slow/integration) avec couverture
	uv run pytest -m "not slow and not integration" --cov --cov-report=term-missing --cov-report=xml

test-all: ## Tous les tests (modèle réel + PostgreSQL de test : DATABASE_URL doit viser une base *_test)
	uv run pytest --cov --cov-report=term-missing

cov: ## Rapport de couverture HTML
	uv run pytest -m "not slow and not integration" --cov --cov-report=html
	@echo "→ htmlcov/index.html"

ci: lint typecheck test ## Enchaînement identique au pipeline CI

# ---------- API ----------
run: ## Lance l'API en local (rechargement auto)
	uv run uvicorn scoring_api.main:app --host 0.0.0.0 --port 8000 --reload

# ---------- Modèle & données ----------
import-model: ## Copie les artefacts MLflow du P6 dans models/
	uv run python scripts/import_model.py --source "$(P6_MODEL_PATH)"

export-model: ## Exporte booster.ubj / model.onnx / model_meta.json
	uv run python scripts/export_model.py

data: ## Construit les échantillons de référence et de production simulée
	uv run python scripts/build_reference_data.py --p6-parquet "$(P6_DATA_PATH)"

# ---------- Docker ----------
docker-build: ## Construit l'image locale
	docker build -t scoring-api:local .

up: ## Démarre la stack complète (api, db, prometheus, grafana, pgadmin)
	$(COMPOSE) --profile tools up -d --build

down: ## Arrête la stack et supprime les volumes
	$(COMPOSE) --profile tools --profile dashboard down -v

logs: ## Logs de l'API
	$(COMPOSE) logs -f api

smoke: ## Test de fumée contre l'API
	bash scripts/smoke_test.sh $(API_URL)

# ---------- Simulation & monitoring ----------
simulate: ## Simule du trafic (SCENARIO=normal|drift_ext_source|drift_credit|drift_young|drift_missing|invalid|mixed)
	uv run python simulation/simulate_traffic.py --url $(API_URL) --scenario $(SCENARIO) --n $(N) --rps $(RPS)

labels: ## Injecte les labels différés pour les requêtes simulées
	uv run python scripts/simulate_labels.py

drift-report: ## Génère le rapport de dérive (Evidently + PSI/KS)
	uv run python -m scoring_api.monitoring.drift --out docs/reports

dashboard: ## Dashboard Streamlit (métier + dérive)
	uv run streamlit run dashboard/app.py

notebook: ## JupyterLab
	uv run jupyter lab notebooks/

# ---------- Optimisation ----------
profile: ## Profilage cProfile de /predict
	uv run python scripts/profile_predict.py

regression: ## Vérifie la non-régression entre backends
	uv run python scripts/check_regression.py --backends xgb_native onnx

bench: ## Benchmark de latence (TAG=nom_de_campagne)
	uv run python benchmarks/bench.py --url $(API_URL) --tag $(TAG) --concurrency 1 8 32

clean: ## Nettoie les caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml profiles
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
