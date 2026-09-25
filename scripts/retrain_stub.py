"""Perspective : ré-entraînement automatique (squelette documenté, non exécuté en production).

Déclencheurs (issus du monitoring) :
  1. dérive du jeu de données (> 30 % des colonnes en alerte PSI) sur une fenêtre glissante ;
  2. coût métier réalisé (labels différés) > référence + 10 % avec ≥ 2 000 labels ;
  3. calendrier (mensuel).

Étapes :
  1. extraire les dossiers labellisés de PostgreSQL (predictions ⋈ prediction_labels, sampled) ;
  2. constituer le jeu d'entraînement = référence historique + nouvelles données labellisées ;
  3. ré-entraîner le pipeline du P6 (mêmes hyperparamètres, seuil optimisé en CV sur le coût) ;
  4. comparer au champion sur un jeu de validation figé (AUC, coût, non-régression, biais) ;
  5. enregistrer dans MLflow, poser l'alias `challenger` ; promotion `champion` manuelle ;
  6. `scripts/import_model.py` + `scripts/export_model.py` + commit → la CI déploie.

Usage (simulation) : uv run python scripts/retrain_stub.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os

from scoring_api.monitoring.reference import load_reference_metrics


def should_retrain(
    cost_realized: float | None, dataset_drift: bool, n_labels: int
) -> tuple[bool, str]:
    ref = load_reference_metrics()["business_cost"]
    if dataset_drift:
        return True, "dérive du jeu de données"
    if cost_realized is not None and n_labels >= 2000 and cost_realized > ref * 1.10:
        return True, f"coût réalisé {cost_realized:.3f} > référence {ref:.3f} + 10 %"
    return False, "aucun déclencheur"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL manquant")

    from scoring_api.monitoring.drift import dataset_summary, drift_table  # noqa: PLC0415
    from scoring_api.monitoring.performance import realized_metrics  # noqa: PLC0415
    from scoring_api.monitoring.production import expand_inputs, load_predictions  # noqa: PLC0415
    from scoring_api.monitoring.reference import load_reference  # noqa: PLC0415

    preds = load_predictions(args.database_url, scenario="normal")
    cur = expand_inputs(preds)
    summary = (
        dataset_summary(drift_table(load_reference(), cur), len(cur))
        if not cur.empty
        else {"dataset_drift": False}
    )
    lab = preds[preds["target"].notna()]
    real = (
        realized_metrics(
            lab["target"].to_numpy(),
            lab["proba_defaut"].to_numpy(),
            float(preds["threshold"].iloc[-1]),
        )
        if len(lab) >= 50
        else {}
    )
    decision, reason = should_retrain(real.get("cout"), bool(summary["dataset_drift"]), len(lab))
    print(
        json.dumps(
            {
                "retrain": decision,
                "reason": reason,
                "n_labels": len(lab),
                "cost_realized": real.get("cout"),
                "dataset_drift": summary["dataset_drift"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if decision and not args.dry_run:
        print("→ ici : extraction, ré-entraînement, évaluation, enregistrement MLflow (challenger)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
