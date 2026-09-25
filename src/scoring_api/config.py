"""Configuration centralisée (variables d'environnement / fichier .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["pyfunc", "xgb_native", "onnx", "fake"]


class Settings(BaseSettings):
    """Paramètres de l'application. Les secrets ne sont jamais écrits en dur."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Application
    app_env: Literal["dev", "test", "prod"] = "dev"
    app_name: str = "pret-a-depenser-scoring-api"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Modèle
    model_path: Path = Path("models/scoring_credit_v1")
    model_backend: ModelBackend = "xgb_native"
    model_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    xgb_nthread: int = Field(default=1, ge=-1)
    # True : /predict s'exécute dans le pool de threads (utile si l'inférence est longue) ;
    # False : exécution inline dans la boucle d'événements (inférence < 1 ms, moins de surcoût).
    predict_in_threadpool: bool = False
    warmup_rows: int = Field(default=10, ge=0)

    # Base de données
    db_enabled: bool = True
    database_url: str | None = None
    db_pool_min: int = Field(default=1, ge=0)
    db_pool_max: int = Field(default=4, ge=1)
    db_auto_migrate: bool = True
    ready_require_db: bool = False

    # Échantillonnage de la journalisation
    log_sample_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    log_grey_zone: float = Field(default=0.05, ge=0.0, le=0.5)
    log_queue_maxsize: int = Field(default=10_000, ge=1)
    log_batch_size: int = Field(default=200, ge=1)
    log_flush_interval_s: float = Field(default=0.5, gt=0)

    # Observabilité
    metrics_enabled: bool = True
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def db_active(self) -> bool:
        return self.db_enabled and bool(self.database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
