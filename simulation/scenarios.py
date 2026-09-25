"""Scénarios de trafic simulé : transformations appliquées aux dossiers de production.

Chaque scénario est une fonction DataFrame → DataFrame, déterministe pour une graine donnée,
afin que les mesures (dérive, latence) soient reproductibles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Scenario = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


def normal(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Trafic conforme à la distribution d'entraînement."""
    return df.copy()


def drift_ext_source(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Dégradation des scores externes (-0,15) : population plus risquée."""
    out = df.copy()
    for c in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
        out[c] = (out[c] - 0.15).clip(lower=0.0, upper=1.0)
    return out


def drift_credit(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inflation des montants : crédits ×1,5, annuités ×1,3 (ratios modifiés)."""
    out = df.copy()
    out["AMT_CREDIT"] = out["AMT_CREDIT"] * 1.5
    out["AMT_ANNUITY"] = out["AMT_ANNUITY"] * 1.3
    out["AMT_GOODS_PRICE"] = out["AMT_GOODS_PRICE"] * 1.4
    return out


def drift_young(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Changement de population : uniquement des clients de moins de 30 ans."""
    young = df[df["DAYS_BIRTH"] > -30 * 365.25]
    return young.sample(
        n=len(df), replace=True, random_state=int(rng.integers(0, 2**31 - 1))
    ).reset_index(drop=True)


def drift_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Panne d'un fournisseur de données : EXT_SOURCE_1 et EXT_SOURCE_3 absents."""
    out = df.copy()
    out["EXT_SOURCE_1"] = np.nan
    out["EXT_SOURCE_3"] = np.nan
    return out


def invalid(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """5 % de dossiers invalides (âge négatif, crédit nul, texte) → erreurs 422 attendues."""
    out = df.copy().astype({"AMT_CREDIT": object})
    n = len(out)
    bad = rng.choice(n, size=max(1, n // 20), replace=False)
    kinds = rng.integers(0, 3, size=len(bad))
    for i, k in zip(bad, kinds, strict=True):
        if k == 0:
            out.loc[out.index[i], "DAYS_BIRTH"] = 1826
        elif k == 1:
            out.loc[out.index[i], "AMT_CREDIT"] = 0
        else:
            out.loc[out.index[i], "AMT_CREDIT"] = "abc"
    return out


def mixed(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mélange : 70 % normal, 20 % scores dégradés, 10 % montants gonflés."""
    n = len(df)
    parts = np.split(
        df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))),
        [int(0.7 * n), int(0.9 * n)],
    )
    return pd.concat(
        [normal(parts[0], rng), drift_ext_source(parts[1], rng), drift_credit(parts[2], rng)]
    ).reset_index(drop=True)


SCENARIOS: dict[str, Scenario] = {
    "normal": normal,
    "drift_ext_source": drift_ext_source,
    "drift_credit": drift_credit,
    "drift_young": drift_young,
    "drift_missing": drift_missing,
    "invalid": invalid,
    "mixed": mixed,
}
