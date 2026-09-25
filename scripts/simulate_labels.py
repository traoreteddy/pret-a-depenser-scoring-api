"""Injecte les labels « différés » (issue réelle du crédit) pour les requêtes simulées.

En production, l'issue d'un crédit n'est connue que des mois plus tard. Ici, les dossiers
simulés proviennent de X_test dont on connaît TARGET : on rattache le label aux prédictions
via client_ref, pour les requêtes plus anciennes que --older-than minutes.

Usage : uv run python scripts/simulate_labels.py [--older-than 0]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import psycopg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--labels", type=Path, default=Path("data/prod_sample_labels.parquet"))
    parser.add_argument("--older-than", type=float, default=0.0, help="minutes")
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL manquant")

    labels = pd.read_parquet(args.labels).set_index("client_ref")["TARGET"].to_dict()
    with psycopg.connect(args.database_url) as conn:
        rows = conn.execute(
            """
            SELECT p.request_id, p.client_ref
            FROM predictions p
            LEFT JOIN prediction_labels l ON l.request_id = p.request_id
            WHERE p.client_ref IS NOT NULL AND l.request_id IS NULL
              AND p.proba_defaut IS NOT NULL
              AND p.ts < now() - (%s * interval '1 minute')
            """,
            (args.older_than,),
        ).fetchall()
        to_insert = [(rid, int(labels[ref])) for rid, ref in rows if ref in labels]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO prediction_labels (request_id, target) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                to_insert,
            )
        conn.commit()
        total = conn.execute("SELECT count(*) FROM prediction_labels").fetchone()
    print(
        f"{len(to_insert)} labels insérés ({len(rows)} candidats) ; total : {total[0] if total else 0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
