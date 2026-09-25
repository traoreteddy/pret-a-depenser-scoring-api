"""Calcul des variables dérivées (port exact de `FeatureEngineer`, notebook P6 n°4, cellule 8).

Deux points d'entrée :
- `build_features(raw)` : une requête API (dict de champs bruts) → vecteur (1, 38) float64 ;
- `engineer_frame(df)` : un DataFrame de champs bruts → DataFrame des 38 variables (référence,
  monitoring), avec exactement les mêmes formules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from scoring_api.features.spec import EPS, FEATURES, N_FEATURES

if TYPE_CHECKING:
    import pandas as pd

_EXT = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
_IDX = {name: i for i, name in enumerate(FEATURES)}


def _f(v: Any) -> float:
    """None → NaN, sinon float."""
    return math.nan if v is None else float(v)


def _std_ddof1(values: list[float]) -> float:
    """Écart-type échantillon (ddof=1) comme `pandas.DataFrame.std(axis=1)` ; NaN si < 2 valeurs."""
    if len(values) < 2:
        return math.nan
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def derived_features(raw: Mapping[str, Any]) -> dict[str, float]:
    """Les 9 variables dérivées à partir des champs bruts (mêmes formules que le notebook)."""
    ext_all = [_f(raw.get(c)) for c in _EXT]
    ext = [v for v in ext_all if not math.isnan(v)]
    amt_credit = _f(raw.get("AMT_CREDIT"))
    days_birth = _f(raw.get("DAYS_BIRTH"))
    return {
        "EXT_SOURCE_MEAN": sum(ext) / len(ext) if ext else math.nan,
        "EXT_SOURCE_MIN": min(ext) if ext else math.nan,
        "EXT_SOURCE_STD": _std_ddof1(ext),
        "EXT_SOURCE_NB_NAN": float(len(ext_all) - len(ext)),
        "CREDIT_ANNUITY_RATIO": amt_credit / (_f(raw.get("AMT_ANNUITY")) + EPS),
        "CREDIT_GOODS_RATIO": amt_credit / (_f(raw.get("AMT_GOODS_PRICE")) + EPS),
        "AGE_ANNEES": -days_birth / 365.25,
        "ID_PUBLISH_AGE_RATIO": _f(raw.get("DAYS_ID_PUBLISH")) / days_birth,
        "PREV_REFUSED_RATIO": _f(raw.get("PREV_REFUSED_COUNT")) / (_f(raw.get("PREV_COUNT")) + EPS),
    }


def build_features(raw: Mapping[str, Any]) -> NDArray[np.float64]:
    """Vecteur (1, 38) dans l'ordre de la signature MLflow. `None` → NaN."""
    derived = derived_features(raw)
    x = np.empty((1, N_FEATURES), dtype=np.float64)
    for name, i in _IDX.items():
        x[0, i] = derived[name] if name in derived else _f(raw.get(name))
    return x


def engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorisée (pandas) : renvoie un DataFrame des 38 variables dans l'ordre."""
    out = df.copy()
    ext = out[list(_EXT)]
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_STD"] = ext.std(axis=1)
    out["EXT_SOURCE_NB_NAN"] = ext.isna().sum(axis=1)
    out["CREDIT_ANNUITY_RATIO"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + EPS)
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + EPS)
    out["AGE_ANNEES"] = -out["DAYS_BIRTH"] / 365.25
    out["ID_PUBLISH_AGE_RATIO"] = out["DAYS_ID_PUBLISH"] / out["DAYS_BIRTH"]
    out["PREV_REFUSED_RATIO"] = out["PREV_REFUSED_COUNT"] / (out["PREV_COUNT"] + EPS)
    return out[list(FEATURES)]
