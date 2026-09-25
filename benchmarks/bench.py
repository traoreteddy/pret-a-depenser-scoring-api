"""Benchmark de latence HTTP reproductible (asyncio + httpx).

Protocole : échauffement, puis N requêtes réelles (dossiers de data/prod_sample.parquet, graine
fixe) à chaque niveau de concurrence ; mesure côté client (aller-retour) et côté serveur
(en-tête X-Process-Time-Ms). Sortie : benchmarks/results/<tag>.json + tableau markdown.

Usage : uv run python benchmarks/bench.py --tag pyfunc --concurrency 1 8 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.simulate_traffic import build_payloads


async def _worker(
    url: str, payloads: list[tuple[str, dict[str, Any]]], concurrency: int, warmup: int
) -> tuple[list[float], list[float], int, float]:
    """Un processus client : `concurrency` connexions keep-alive, mesure aller-retour et serveur."""
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    client_ms: list[float] = []
    server_ms: list[float] = []
    errors = 0
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=30.0) as client:
        sem = asyncio.Semaphore(concurrency)

        async def one(body: dict[str, Any], record: bool) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/predict", json=body)
                    dt = (time.perf_counter() - t0) * 1000
                    if record:
                        if r.status_code == 200:
                            client_ms.append(dt)
                            server_ms.append(float(r.headers.get("x-process-time-ms", "nan")))
                        else:
                            errors += 1
                except httpx.HTTPError:
                    if record:
                        errors += 1

        await asyncio.gather(*(one(b, False) for _, b in payloads[:warmup]))
        t0 = time.perf_counter()
        await asyncio.gather(*(one(b, True) for _, b in payloads))
        elapsed = time.perf_counter() - t0
    return client_ms, server_ms, errors, elapsed


def _worker_entry(
    args: tuple[str, list[tuple[str, dict[str, Any]]], int, int],
) -> tuple[list[float], list[float], int, float]:
    return asyncio.run(_worker(*args))


def run_level(
    url: str, payloads: list[tuple[str, dict[str, Any]]], concurrency: int, warmup: int, procs: int
) -> dict[str, Any]:
    """Répartit la charge sur `procs` processus clients pour que le générateur ne soit pas le goulot
    (un seul processus Python sature vers 1 000 req/s et fausse le p99 au-delà de 8 connexions)."""
    procs = max(1, min(procs, concurrency))
    per_proc = max(1, concurrency // procs)
    chunks = [payloads[i::procs] for i in range(procs)]
    jobs = [(url, chunk, per_proc, max(1, warmup // procs)) for chunk in chunks]
    if procs == 1:
        results = [_worker_entry(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            results = pool.map(_worker_entry, jobs)
    client_ms = [v for r in results for v in r[0]]
    server_ms = [v for r in results for v in r[1]]
    errors = sum(r[2] for r in results)
    elapsed = max(r[3] for r in results)
    c = np.array(client_ms)
    s = np.array(server_ms)
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else None  # noqa: E731
    return {
        "concurrency": concurrency,
        "procs": procs,
        "n": len(payloads),
        "errors": errors,
        "rps": round(len(payloads) / elapsed, 1),
        "client_ms": {
            "p50": q(c, 50),
            "p95": q(c, 95),
            "p99": q(c, 99),
            "max": round(float(c.max()), 2) if len(c) else None,
        },
        "server_ms": {"p50": q(s, 50), "p95": q(s, 95), "p99": q(s, 99)},
    }


def server_info(url: str) -> dict[str, Any]:
    try:
        r = httpx.get(f"{url}/health/ready", timeout=5)
        return dict(r.json().get("model") or {})
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--repeat", type=int, default=3, help="répétitions par niveau (médiane du p99 retenue)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--procs", type=int, default=4, help="processus clients (répartition de la concurrence)"
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    payloads = build_payloads(Path("data/prod_sample.parquet"), "normal", args.n, args.seed)
    levels = []
    for c in args.concurrency:
        runs = [
            run_level(args.url, payloads, c, args.warmup, args.procs) for _ in range(args.repeat)
        ]
        runs.sort(key=lambda r: r["client_ms"]["p99"] or 0)
        median = runs[len(runs) // 2]
        median["runs_p99_client"] = [r["client_ms"]["p99"] for r in runs]
        levels.append(median)
        print(
            f"c={c:>3}  rps={median['rps']:>7}  client p50/p95/p99 = {median['client_ms']['p50']}/{median['client_ms']['p95']}/{median['client_ms']['p99']} ms  server p99 = {median['server_ms']['p99']} ms  err={median['errors']}"
        )

    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        chip = ""
    result = {
        "tag": args.tag,
        "note": args.note,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": args.url,
        "model": server_info(args.url),
        "protocol": {
            "n": args.n,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "payloads": "data/prod_sample.parquet (scénario normal)",
        },
        "host": {
            "platform": platform.platform(),
            "chip": chip or platform.processor(),
            "python": platform.python_version(),
        },
        "levels": levels,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    lines = [
        f"| {args.tag} | "
        + " | ".join(f"{lv['client_ms']['p50']} / {lv['client_ms']['p99']}" for lv in levels)
        + " | "
        + " / ".join(str(lv["server_ms"]["p99"]) for lv in levels)
        + " |"
    ]
    print("\nmarkdown (client p50 / p99 par concurrence ; serveur p99) :")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
