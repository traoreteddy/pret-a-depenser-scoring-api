"""Spécification des variables du modèle `scoring_credit` v1.

Source de vérité : signature MLflow du modèle (`models/scoring_credit_v1/MLmodel`) et
classe `FeatureEngineer` du notebook P6 `4_modelisation.ipynb` (cellule 8).
"""

from __future__ import annotations

from typing import Final

# Les 38 variables attendues par le modèle, dans l'ordre exact de la signature MLflow.
FEATURES: Final[tuple[str, ...]] = (
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "CREDIT_ANNUITY_RATIO",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "AMT_GOODS_PRICE",
    "INST_LATE_RATIO",
    "CREDIT_GOODS_RATIO",
    "INST_AMT_PAYMENT_SUM",
    "PREV_CREDIT_APP_RATIO_MEAN",
    "CC_UTIL_MAX",
    "BUREAU_ACTIVE_RATIO",
    "POS_INSTALMENT_FUTURE_MEAN",
    "PREV_REFUSED_RATIO",
    "CC_UTIL_MEAN",
    "DAYS_ID_PUBLISH",
    "BUREAU_DAYS_CREDIT_MIN",
    "AGE_ANNEES",
    "BUREAU_DAYS_CREDIT_MAX",
    "EXT_SOURCE_NB_NAN",
    "PREV_AMT_CREDIT_MAX",
    "FLAG_DOCUMENT_3",
    "PREV_APPROVED_RATIO",
    "OWN_CAR_AGE",
    "INST_COUNT",
    "PREV_AMT_CREDIT_MEAN",
    "FLOORSMAX_AVG",
    "DAYS_LAST_PHONE_CHANGE",
    "EXT_SOURCE_STD",
    "FLAG_EMP_PHONE",
    "PREV_CNT_PAYMENT_MEAN",
    "PREV_COUNT",
    "ID_PUBLISH_AGE_RATIO",
    "ELEVATORS_AVG",
    "LIVINGAREA_AVG",
    "PREV_AMT_ANNUITY_MEAN",
)

# Variables dérivées, calculées par l'API à partir des champs bruts (jamais fournies par le client).
ENGINEERED: Final[tuple[str, ...]] = (
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_STD",
    "EXT_SOURCE_NB_NAN",
    "CREDIT_ANNUITY_RATIO",
    "CREDIT_GOODS_RATIO",
    "AGE_ANNEES",
    "ID_PUBLISH_AGE_RATIO",
    "PREV_REFUSED_RATIO",
)

# Champs bruts nécessaires aux dérivées mais absents de la signature.
RAW_ONLY: Final[tuple[str, ...]] = ("AMT_CREDIT", "DAYS_BIRTH", "PREV_REFUSED_COUNT")

# Les 32 champs bruts attendus dans la requête API.
RAW_FIELDS: Final[tuple[str, ...]] = tuple(f for f in FEATURES if f not in ENGINEERED) + RAW_ONLY

# Colonnes typées « long » dans la signature : ne doivent jamais être NaN.
INT_FEATURES: Final[frozenset[str]] = frozenset(
    {"DAYS_ID_PUBLISH", "EXT_SOURCE_NB_NAN", "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE"}
)

EPS: Final[float] = 1e-6
N_FEATURES: Final[int] = len(FEATURES)

assert N_FEATURES == 38
assert len(RAW_FIELDS) == 32
assert len(set(RAW_FIELDS)) == 32
