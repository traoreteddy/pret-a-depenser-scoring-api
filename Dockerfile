# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Étape 1 : construction de l'environnement virtuel avec uv (verrouillé par uv.lock)
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Extras à installer : vide = image allégée (backend natif xgb_native, défaut de production) ;
# "pyfunc" = baseline MLflow ; "pyfunc onnx" = tous les backends (benchmarks)
ARG INSTALL_EXTRAS=""

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    extras=""; for e in $INSTALL_EXTRAS; do extras="$extras --extra $e"; done; \
    uv sync --frozen --no-dev --no-install-project $extras

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    extras=""; for e in $INSTALL_EXTRAS; do extras="$extras --extra $e"; done; \
    uv sync --frozen --no-dev $extras

# ---------------------------------------------------------------------------
# Étape 2 : image d'exécution minimale
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="pret-a-depenser-scoring-api" \
      org.opencontainers.image.source="https://github.com/traoreteddy/pret-a-depenser-scoring-api" \
      org.opencontainers.image.description="API de scoring crédit (FastAPI + XGBoost)"

# libgomp1 : OpenMP requis par XGBoost ; curl : healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    APP_ENV=prod \
    MODEL_PATH=/app/models/scoring_credit_v1 \
    PORT=8000 \
    WEB_CONCURRENCY=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health/live || exit 1

CMD ["sh", "-c", "uvicorn scoring_api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --no-access-log"]
