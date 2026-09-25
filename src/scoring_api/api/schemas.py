"""Contrats d'entrée/sortie de l'API (validation stricte des données client)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bornes de plausibilité (documentées dans le README). Elles couvrent 100 % des valeurs
# observées dans les données d'entraînement P6 et rejettent les valeurs aberrantes.
Montant = Annotated[float, Field(gt=0, le=1e8, description="Montant en unités monétaires")]
MontantOpt = Annotated[float | None, Field(gt=0, le=1e8)]
Ratio01 = Annotated[float | None, Field(ge=0, le=1)]
Positif = Annotated[float | None, Field(ge=0, le=1e9)]
JoursNeg = Annotated[float | None, Field(ge=-50_000, le=0, description="Jours relatifs (≤ 0)")]
# `int` strict refuse les booléens (True ≠ 1) contrairement à Literal[0, 1].
Flag = Annotated[int, Field(ge=0, le=1, description="Indicateur binaire 0/1")]


class ClientInput(BaseModel):
    """Données brutes d'un dossier client (32 champs). Les 9 variables dérivées sont calculées par l'API.

    Mode strict : les types doivent être exacts (pas de "30000" pour un nombre) et tout champ
    inconnu est refusé. `null` est accepté pour les champs optionnels (valeur manquante).
    """

    model_config = ConfigDict(strict=True, extra="forbid", json_schema_extra={"examples": []})

    # --- Dossier de crédit ---
    AMT_CREDIT: Montant = Field(description="Montant du crédit demandé")
    AMT_ANNUITY: MontantOpt = Field(default=None, description="Annuité du crédit")
    AMT_GOODS_PRICE: MontantOpt = Field(default=None, description="Prix du bien financé")

    # --- Identité ---
    DAYS_BIRTH: int = Field(
        ge=-36_525, le=-6_570, description="Âge en jours négatifs (18 à 100 ans)"
    )
    DAYS_ID_PUBLISH: int = Field(
        ge=-36_525, le=0, description="Jours depuis l'émission de la pièce d'identité"
    )
    DAYS_LAST_PHONE_CHANGE: JoursNeg = None
    FLAG_DOCUMENT_3: Flag
    FLAG_EMP_PHONE: Flag
    OWN_CAR_AGE: Annotated[float | None, Field(ge=0, le=100)] = None

    # --- Scores externes ---
    EXT_SOURCE_1: Ratio01 = None
    EXT_SOURCE_2: Ratio01 = None
    EXT_SOURCE_3: Ratio01 = None

    # --- Logement (normalisé 0-1) ---
    FLOORSMAX_AVG: Ratio01 = None
    ELEVATORS_AVG: Ratio01 = None
    LIVINGAREA_AVG: Ratio01 = None

    # --- Bureau de crédit ---
    BUREAU_ACTIVE_RATIO: Ratio01 = None
    BUREAU_DAYS_CREDIT_MIN: JoursNeg = None
    BUREAU_DAYS_CREDIT_MAX: JoursNeg = None

    # --- Demandes précédentes ---
    PREV_COUNT: Positif = None
    PREV_REFUSED_COUNT: Positif = None
    PREV_APPROVED_RATIO: Ratio01 = None
    PREV_AMT_CREDIT_MEAN: Positif = None
    PREV_AMT_CREDIT_MAX: Positif = None
    PREV_AMT_ANNUITY_MEAN: Positif = None
    PREV_CREDIT_APP_RATIO_MEAN: Positif = None
    PREV_CNT_PAYMENT_MEAN: Positif = None

    # --- Échéances, cartes, POS ---
    INST_COUNT: Positif = None
    INST_LATE_RATIO: Ratio01 = None
    INST_AMT_PAYMENT_SUM: Positif = None
    CC_UTIL_MEAN: Annotated[float | None, Field(ge=-1, le=100)] = None
    CC_UTIL_MAX: Annotated[float | None, Field(ge=-1, le=100)] = None
    POS_INSTALMENT_FUTURE_MEAN: Positif = None

    @model_validator(mode="after")
    def _coherence(self) -> ClientInput:
        if (
            self.PREV_REFUSED_COUNT is not None
            and self.PREV_COUNT is not None
            and self.PREV_REFUSED_COUNT > self.PREV_COUNT
        ):
            msg = "PREV_REFUSED_COUNT ne peut pas dépasser PREV_COUNT"
            raise ValueError(msg)
        if (
            self.BUREAU_DAYS_CREDIT_MIN is not None
            and self.BUREAU_DAYS_CREDIT_MAX is not None
            and self.BUREAU_DAYS_CREDIT_MIN > self.BUREAU_DAYS_CREDIT_MAX
        ):
            msg = "BUREAU_DAYS_CREDIT_MIN doit être ≤ BUREAU_DAYS_CREDIT_MAX"
            raise ValueError(msg)
        return self


class PredictionResponse(BaseModel):
    request_id: str
    proba_defaut: float = Field(ge=0, le=1, description="Probabilité de défaut de paiement")
    decision: Literal["Accordé", "Refusé"]
    threshold: float = Field(description="Seuil métier appliqué (refus si proba ≥ seuil)")
    model_name: str
    model_version: str
    model_backend: str
    latency_ms: float = Field(description="Temps de traitement serveur (features + inférence)")


class ErrorDetail(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(BaseModel):
    request_id: str
    error: str = Field(description="validation_error | model_error | internal_error | not_ready")
    message: str
    details: list[ErrorDetail] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "not_ready"]
    model_loaded: bool
    model: dict[str, str | float] | None = None
    db: Literal["ok", "degraded", "disabled", "unknown"] = "unknown"
    version: str


EXAMPLE_CLIENT: dict[str, float | int | None] = {
    "AMT_CREDIT": 513_000.0,
    "AMT_ANNUITY": 24_903.0,
    "AMT_GOODS_PRICE": 450_000.0,
    "DAYS_BIRTH": -14_000,
    "DAYS_ID_PUBLISH": -2_500,
    "DAYS_LAST_PHONE_CHANGE": -600.0,
    "FLAG_DOCUMENT_3": 1,
    "FLAG_EMP_PHONE": 1,
    "OWN_CAR_AGE": None,
    "EXT_SOURCE_1": None,
    "EXT_SOURCE_2": 0.62,
    "EXT_SOURCE_3": 0.55,
    "FLOORSMAX_AVG": 0.1667,
    "ELEVATORS_AVG": 0.0,
    "LIVINGAREA_AVG": 0.08,
    "BUREAU_ACTIVE_RATIO": 0.5,
    "BUREAU_DAYS_CREDIT_MIN": -1_800.0,
    "BUREAU_DAYS_CREDIT_MAX": -300.0,
    "PREV_COUNT": 3.0,
    "PREV_REFUSED_COUNT": 0.0,
    "PREV_APPROVED_RATIO": 1.0,
    "PREV_AMT_CREDIT_MEAN": 120_000.0,
    "PREV_AMT_CREDIT_MAX": 200_000.0,
    "PREV_AMT_ANNUITY_MEAN": 9_500.0,
    "PREV_CREDIT_APP_RATIO_MEAN": 1.05,
    "PREV_CNT_PAYMENT_MEAN": 12.0,
    "INST_COUNT": 30.0,
    "INST_LATE_RATIO": 0.05,
    "INST_AMT_PAYMENT_SUM": 250_000.0,
    "CC_UTIL_MEAN": None,
    "CC_UTIL_MAX": None,
    "POS_INSTALMENT_FUTURE_MEAN": 6.0,
}
ClientInput.model_config["json_schema_extra"] = {"examples": [dict(EXAMPLE_CLIENT)]}  # type: ignore[dict-item]
