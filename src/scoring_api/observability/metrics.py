"""Métriques Prometheus de l'API (latence, volumes, erreurs, distribution des scores)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from scoring_api.model.base import ModelInfo

# Bucket 0.18 s : lecture directe de la SLO « p99 < 180 ms ».
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.18, 0.25, 0.5, 1.0, 2.5)
INFER_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
SCORE_BUCKETS = tuple(round(i * 0.05, 2) for i in range(1, 21))

http_requests = Counter("http_requests_total", "Requêtes HTTP", ["method", "path", "status"])
http_duration = Histogram(
    "http_request_duration_seconds", "Durée des requêtes HTTP", ["path"], buckets=HTTP_BUCKETS
)
inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Durée de l'inférence modèle",
    ["backend"],
    buckets=INFER_BUCKETS,
)
features_duration = Histogram(
    "feature_engineering_duration_seconds", "Durée du calcul des variables", buckets=INFER_BUCKETS
)
predictions = Counter("predictions_total", "Prédictions par décision", ["decision"])
prediction_errors = Counter("prediction_errors_total", "Erreurs de prédiction", ["error_type"])
prediction_score = Histogram(
    "prediction_score", "Distribution des probabilités de défaut", buckets=SCORE_BUCKETS
)
log_queue_size = Gauge("prediction_log_queue_size", "Taille de la file de journalisation")
log_dropped = Counter(
    "prediction_log_dropped_total", "Entrées de journal abandonnées (file pleine)"
)
log_batches = Counter("prediction_log_batches_total", "Lots insérés en base")
log_db_errors = Counter("prediction_log_db_errors_total", "Erreurs d'insertion en base")
model_info = Gauge("model_info", "Modèle chargé", ["name", "version", "backend", "threshold"])
app_ready = Gauge("app_ready", "1 si l'API est prête")

_PATH_ALLOWLIST = {
    "/predict",
    "/health/live",
    "/health/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
}


def _norm_path(path: str) -> str:
    return path if path in _PATH_ALLOWLIST else "other"


def on_response(method: str, path: str, status: int, elapsed_s: float) -> None:
    p = _norm_path(path)
    http_requests.labels(method=method, path=p, status=str(status)).inc()
    http_duration.labels(path=p).observe(elapsed_s)


def observe_prediction(
    backend: str, proba: float, decision: str, fe_ms: float, inf_ms: float
) -> None:
    features_duration.observe(fe_ms / 1000)
    inference_duration.labels(backend=backend).observe(inf_ms / 1000)
    predictions.labels(decision=decision).inc()
    prediction_score.observe(proba)


def set_model_info(info: ModelInfo) -> None:
    model_info.labels(
        name=info.name,
        version=info.version,
        backend=info.backend,
        threshold=f"{info.threshold:.4f}",
    ).set(1)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
