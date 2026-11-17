"""Lightweight benchmark harness for the perfusion engine.

Times the core solves at a few grid sizes and prints a small table — a quick way to
spot performance regressions and compare the direct vs. AMG Darcy backends. Run with
``uv run python scripts/benchmark.py`` (add ``--extra amg`` synced for the AMG row).

This is intentionally dependency-free (``time.perf_counter``); for tracking across
commits, pipe the output into your tool of choice. Not a pytest — it is a script.
"""

from __future__ import annotations

import time

import numpy as np

from perfusion import Grid, Parameters, run_hybrid
from perfusion.core.darcy import tpfa_operator
from perfusion.core.linsolve import amg_available, solve_spd


def _time(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_hybrid(sizes=((60, 70), (120, 140), (200, 220))) -> list[dict]:
    rows = []
    prm = Parameters(r_min_frac=0.2, seed=0)
    for nx, ny in sizes:
        grid = Grid(nx=nx, ny=ny, lx=140, ly=160)
        dt = _time(lambda g=grid: run_hybrid(prm, g, domain_limit=(-70, 70, 0, 160)))
        rows.append({"solve": "run_hybrid", "grid": f"{nx}x{ny}", "seconds": round(dt, 4)})
    return rows


def bench_darcy_backends(n: int = 120) -> list[dict]:
    grid = Grid(nx=n, ny=n)
    a = tpfa_operator(grid)[0].tolil()
    a[0, :] = 0
    a[0, 0] = 1
    a = a.tocsr()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(a.shape[0])
    b[0] = 0.0
    rows = [{"solve": "darcy:direct", "grid": f"{n}x{n}",
             "seconds": round(_time(lambda: solve_spd(a, b, method="direct")), 4)}]
    if amg_available():
        rows.append({"solve": "darcy:amg", "grid": f"{n}x{n}",
                     "seconds": round(_time(lambda: solve_spd(a, b, method="amg")), 4)})
    return rows


def main() -> None:
    rows = bench_hybrid() + bench_darcy_backends()
    width = max(len(r["solve"]) for r in rows)
    print(f"{'solve':<{width}}  {'grid':>10}  {'seconds':>9}")
    for r in rows:
        print(f"{r['solve']:<{width}}  {r['grid']:>10}  {r['seconds']:>9.4f}")


if __name__ == "__main__":
    main()
