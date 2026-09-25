"""Agrège benchmarks/results/*.json en un tableau markdown (client p50 / p99 et serveur p99 par concurrence).

Usage : uv run python benchmarks/report.py [--tags a b c]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()
    files = sorted(args.dir.glob("*.json"))
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        if "levels" not in d or (args.tags and d["tag"] not in args.tags):
            continue
        rows.append(d)
    if not rows:
        print("aucun résultat")
        return 1
    levels = [lv["concurrency"] for lv in rows[0]["levels"]]
    head = (
        "| Configuration | "
        + " | ".join(f"c={c} client p50 / p99 (ms)" for c in levels)
        + " | "
        + " | ".join(f"c={c} serveur p99" for c in levels)
        + " | RPS max |"
    )
    sep = "|---|" + "---|" * (2 * len(levels)) + "---|"
    print(head)
    print(sep)
    for d in rows:
        cells = [f"{lv['client_ms']['p50']} / **{lv['client_ms']['p99']}**" for lv in d["levels"]]
        srv = [f"{lv['server_ms']['p99']}" for lv in d["levels"]]
        rps = max(lv["rps"] for lv in d["levels"])
        print(
            f"| `{d['tag']}` — {d['note']} | "
            + " | ".join(cells)
            + " | "
            + " | ".join(srv)
            + f" | {rps:.0f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
