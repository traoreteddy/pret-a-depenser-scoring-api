"""Construit les jeux de données du monitoring à partir du parquet du P6 (jamais les CSV bruts).

- data/reference_sample.parquet : échantillon de X_test (champs bruts + 38 variables + TARGET +
  proba du modèle) → référence pour la dérive et le calibrage du seuil ;
- data/prod_sample.parquet : lignes disjointes de X_test, champs bruts uniquement + client_ref →
  trafic de production simulé ;
- data/prod_sample_labels.parquet : TARGET des lignes de prod_sample (labels « différés ») ;
- data/reference_metrics.json : métriques du modèle sur tout X_test (AUC, coût métier, refus).

Le découpage train/test reproduit exactement celui du notebook P6 n°4 :
train_test_split(test_size=0.2, stratify=y, random_state=42).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scoring_api.config import Settings
from scoring_api.features.engineering import engineer_frame
from scoring_api.features.spec import FEATURES, RAW_FIELDS
from scoring_api.model.registry import load_predictor

COST_FN, COST_FP = 10, 1


def business_cost(y: np.ndarray, proba: np.ndarray, threshold: float) -> float:  # type: ignore[type-arg]
    pred = (proba >= threshold).astype(int)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return (COST_FN * fn + COST_FP * fp) / len(y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p6-parquet", required=True, type=Path)
    parser.add_argument("--out", default=Path("data"), type=Path)
    parser.add_argument("--n-reference", type=int, default=20_000)
    parser.add_argument("--n-prod", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.p6_parquet)
    y = df["TARGET"]
    redondantes = [c for c in df if c.endswith(("_MODE", "_MEDI")) and c != "TOTALAREA_MODE"]
    x_raw = df.drop(columns=["TARGET", *redondantes], errors="ignore")
    _, x_test, _, y_test = train_test_split(x_raw, y, test_size=0.2, stratify=y, random_state=42)
    print(f"X_test : {x_test.shape}")

    raw = x_test[list(RAW_FIELDS)].copy()
    feats = engineer_frame(x_test)
    predictor = load_predictor(Settings(_env_file=None, model_backend="pyfunc", db_enabled=False))  # type: ignore[call-arg]
    proba = predictor.predict_proba(feats.to_numpy(dtype=np.float64))
    threshold = predictor.info.threshold
    yt = y_test.to_numpy()

    metrics = {
        "n_test": len(yt),
        "threshold": threshold,
        "auc": float(roc_auc_score(yt, proba)),
        "business_cost": business_cost(yt, proba, threshold),
        "business_cost_accept_all": float(COST_FN * yt.mean()),
        "refusal_rate": float((proba >= threshold).mean()),
        "default_rate": float(yt.mean()),
        "proba_mean": float(proba.mean()),
        "proba_quantiles": {
            str(q): float(np.quantile(proba, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "split": {"test_size": 0.2, "stratify": True, "random_state": 42},
        "source": str(args.p6_parquet),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in {"proba_quantiles", "split", "source"}},
            indent=2,
        )
    )

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(x_test))
    ref_idx, prod_idx = (
        idx[: args.n_reference],
        idx[args.n_reference : args.n_reference + args.n_prod],
    )

    reference = raw.iloc[ref_idx].reset_index(drop=True)
    for f in FEATURES:
        if f not in reference:
            reference[f] = feats.iloc[ref_idx][f].to_numpy()
    reference["TARGET"] = yt[ref_idx]
    reference["proba_defaut"] = proba[ref_idx]
    reference.to_parquet(args.out / "reference_sample.parquet", index=False, compression="zstd")

    prod = raw.iloc[prod_idx].reset_index(drop=True)
    prod.insert(0, "client_ref", [f"sim-{i:05d}" for i in range(len(prod))])
    prod.to_parquet(args.out / "prod_sample.parquet", index=False, compression="zstd")
    labels = pd.DataFrame(
        {
            "client_ref": prod["client_ref"],
            "TARGET": yt[prod_idx],
            "proba_reference": proba[prod_idx],
        }
    )
    labels.to_parquet(args.out / "prod_sample_labels.parquet", index=False, compression="zstd")

    for f in sorted(args.out.glob("*.parquet")):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:6.2f} Mo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
