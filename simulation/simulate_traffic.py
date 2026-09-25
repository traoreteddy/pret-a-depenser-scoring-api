"""Simulateur de trafic : envoie des dossiers de production (réels, tirés de X_test) à l'API.

Chaque requête porte `X-Client-Ref` (pour rattacher le label différé) et `X-Scenario` (pour
isoler le scénario dans les analyses). Le résumé (codes HTTP, latences client et serveur) est
écrit dans simulation/results/<scenario>_<horodatage>.json.

Usage : uv run python simulation/simulate_traffic.py --scenario drift_ext_source --n 2000 --rps 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.scenarios import SCENARIOS

INT_FIELDS = ("DAYS_BIRTH", "DAYS_ID_PUBLISH", "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE")


def to_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "client_ref":
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif k in INT_FIELDS and isinstance(v, int | float | np.integer | np.floating):
            out[k] = int(v)
        elif isinstance(v, np.integer | np.floating):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def build_payloads(
    sample: Path, scenario: str, n: int, seed: int
) -> list[tuple[str, dict[str, Any]]]:
    df = pd.read_parquet(sample)
    rng = np.random.default_rng(seed)
    if n > len(df):
        df = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df = SCENARIOS[scenario](df, rng)
    return [(str(r["client_ref"]), to_payload(r)) for r in df.to_dict(orient="records")]


async def run(
    url: str,
    payloads: list[tuple[str, dict[str, Any]]],
    scenario: str,
    rps: float,
    concurrency: int,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    results: list[tuple[int, float, float | None]] = []
    interval = 1.0 / rps if rps > 0 else 0.0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=10.0) as client:

        async def one(ref: str, body: dict[str, Any]) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(
                        "/predict", json=body, headers={"X-Client-Ref": ref, "X-Scenario": scenario}
                    )
                    server = r.headers.get("x-process-time-ms")
                    results.append(
                        (
                            r.status_code,
                            (time.perf_counter() - t0) * 1000,
                            float(server) if server else None,
                        )
                    )
                except httpx.HTTPError:
                    results.append((0, (time.perf_counter() - t0) * 1000, None))

        start = time.perf_counter()
        tasks = []
        for i, (ref, body) in enumerate(payloads):
            if interval:
                target = start + i * interval
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(ref, body)))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    codes = [c for c, _, _ in results]
    client_ms = np.array([t for _, t, _ in results])
    server_ms = np.array([s for _, _, s in results if s is not None])
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None  # noqa: E731
    return {
        "scenario": scenario,
        "url": url,
        "n": len(results),
        "duration_s": round(elapsed, 2),
        "achieved_rps": round(len(results) / elapsed, 1),
        "status": {str(c): codes.count(c) for c in sorted(set(codes))},
        "client_latency_ms": {
            "p50": pct(client_ms, 50),
            "p95": pct(client_ms, 95),
            "p99": pct(client_ms, 99),
            "max": float(client_ms.max()),
        },
        "server_latency_ms": {
            "p50": pct(server_ms, 50),
            "p95": pct(server_ms, 95),
            "p99": pct(server_ms, 99),
        },
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rps", type=float, default=50.0, help="0 = sans limitation")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=Path, default=Path("data/prod_sample.parquet"))
    parser.add_argument("--out", type=Path, default=Path("simulation/results"))
    args = parser.parse_args()

    payloads = build_payloads(args.sample, args.scenario, args.n, args.seed)
    summary = asyncio.run(run(args.url, payloads, args.scenario, args.rps, args.concurrency))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (args.out / f"{args.scenario}_{stamp}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return 0 if summary["status"].get("0", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
