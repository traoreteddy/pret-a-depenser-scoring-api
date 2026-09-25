"""Détection de dérive des données : PSI, Kolmogorov-Smirnov et rapport Evidently.

Seuils explicites (documentés dans le README) :
- PSI < 0,10 : stable ; 0,10 ≤ PSI < 0,25 : à surveiller ; PSI ≥ 0,25 : dérive ;
- KS : p-value < alpha / n_colonnes (correction de Bonferroni) ;
- dérive « dataset » si plus de `share` (30 %) des colonnes sont en alerte PSI (≥ 0,10).
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scoring_api.features.spec import ENGINEERED, RAW_FIELDS

PSI_WARN, PSI_ALERT = 0.10, 0.25
MIN_ROWS = 200
MONITORED_COLUMNS: tuple[str, ...] = (*RAW_FIELDS, *ENGINEERED, "proba_defaut")


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index sur des quantiles de la référence (les NaN forment un bin à part)."""
    ref = reference.astype(float)
    cur = current.astype(float)
    ref_nn, cur_nn = ref.dropna(), cur.dropna()
    if len(ref_nn) < 10 or len(cur_nn) < 10:
        return float("nan")
    edges = np.unique(np.quantile(ref_nn, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variable quasi constante : comparer les fréquences des valeurs
        edges = np.unique(np.concatenate([edges, edges + 1e-9]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref_nn, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur_nn, bins=edges)[0].astype(float)
    ref_counts = np.append(ref_counts, ref.isna().sum())
    cur_counts = np.append(cur_counts, cur.isna().sum())
    ref_p = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_status(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value >= PSI_ALERT:
        return "dérive"
    if value >= PSI_WARN:
        return "à surveiller"
    return "stable"


def drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = MONITORED_COLUMNS,
    alpha: float = 0.01,
) -> pd.DataFrame:
    cols = [c for c in columns if c in reference.columns and c in current.columns]
    rows: list[dict[str, Any]] = []
    for c in cols:
        ref, cur = reference[c].astype(float), current[c].astype(float)
        ks_stat, ks_p = (np.nan, np.nan)
        if ref.notna().sum() >= 10 and cur.notna().sum() >= 10:
            ks_stat, ks_p = stats.ks_2samp(ref.dropna(), cur.dropna())
        value = psi(ref, cur)
        rows.append(
            {
                "colonne": c,
                "type": "dérivée"
                if c in ENGINEERED
                else ("score" if c == "proba_defaut" else "brute"),
                "psi": value,
                "statut_psi": psi_status(value),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "ks_derive": bool(ks_p < alpha / max(len(cols), 1))
                if not np.isnan(ks_p)
                else False,
                "moy_ref": float(ref.mean()),
                "moy_cur": float(cur.mean()),
                "nan_ref": float(ref.isna().mean()),
                "nan_cur": float(cur.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def dataset_summary(table: pd.DataFrame, n_current: int, share: float = 0.30) -> dict[str, Any]:
    valid = table[table["statut_psi"] != "n/a"]
    n_alert = int((valid["psi"] >= PSI_WARN).sum())
    n_drift = int((valid["psi"] >= PSI_ALERT).sum())
    n_ks = int(valid["ks_derive"].sum())
    ratio = n_alert / max(len(valid), 1)
    return {
        "n_current": n_current,
        "n_columns": len(valid),
        "n_psi_warn": n_alert,
        "n_psi_drift": n_drift,
        "n_ks_drift": n_ks,
        "share_alert": ratio,
        "dataset_drift": bool(ratio > share and n_current >= MIN_ROWS),
        "sufficient_data": n_current >= MIN_ROWS,
        "score_psi": float(valid.loc[valid["colonne"] == "proba_defaut", "psi"].iloc[0])
        if (valid["colonne"] == "proba_defaut").any()
        else float("nan"),
    }


def evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    out_html: Path | None = None,
) -> Any:
    """Rapport Evidently (DataDriftPreset). Renvoie le snapshot ; écrit le HTML si demandé."""
    from evidently import Report  # noqa: PLC0415
    from evidently.presets import DataDriftPreset  # noqa: PLC0415

    cols = [
        c
        for c in (columns or list(MONITORED_COLUMNS))
        if c in reference.columns and c in current.columns
    ]
    report = Report([DataDriftPreset(columns=cols, drift_share=0.3)])
    snapshot = report.run(current[cols].astype(float), reference[cols].astype(float))
    if out_html is not None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out_html))
    return snapshot


def refusal_rate_ci(n_refused: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Taux de refus et intervalle de confiance binomial (Wilson)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = n_refused / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - half, centre + half


def main() -> int:
    from scoring_api.monitoring.production import expand_inputs, load_predictions  # noqa: PLC0415
    from scoring_api.monitoring.reference import load_reference  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Rapport de dérive (PSI/KS + Evidently) sur les données de production"
    )
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/reports"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL manquant")

    since = datetime.now(UTC) - timedelta(hours=args.since_hours)
    preds = load_predictions(args.database_url, since=since, scenario=args.scenario)
    current = expand_inputs(preds)
    reference = load_reference()
    if current.empty:
        print("Aucune donnée échantillonnée sur la période.")
        return 1
    table = drift_table(reference, current)
    summary = dataset_summary(table, len(current))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = args.scenario or "all"
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"drift_{tag}_{stamp}.csv", index=False)
    evidently_report(reference, current, out_html=args.out / f"evidently_{tag}_{stamp}.html")
    pd.set_option("display.width", 160)
    print(table.head(15).round(4).to_string(index=False))
    print("\nRésumé :", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
