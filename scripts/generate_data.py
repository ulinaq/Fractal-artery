"""Generate the committed data artifacts used by examples and regression tests.

Run with ``uv run python scripts/generate_data.py``. Writes:

- ``data/synthetic_perfusion.npz`` -- a small, reusable DCE-MRI-like dataset
  (time grid, arterial input function, and tissue curves for several known
  ``(Fp, vp)`` ground-truth values). Handy for tracer-kinetic-model demos/tests.
- ``tests/data/reference_hybrid.npz`` -- a locked reference solution of the
  hybrid model (pressure/flux/TOF summaries + trunk flows) so numerical
  regressions are caught.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import Grid, Parameters, run_hybrid, tkm

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TEST_DATA = REPO / "tests" / "data"


def make_synthetic_perfusion() -> None:
    t = np.arange(0.0, 120.0, 0.5)
    aif = tkm.gamma_variate_aif(t, t0=5.0, A=3.0, B=1.5)
    # ground-truth perfusion parameters for a few "tissues"
    ground_truth = np.array([(40.0, 8.0), (20.0, 5.0), (60.0, 12.0)])
    curves = np.stack([tkm.one_compartment(t, aif, Fp, vp)
                       for Fp, vp in ground_truth])
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "synthetic_perfusion.npz",
             t=t, aif=aif, ground_truth=ground_truth, tissue_curves=curves)
    print(f"wrote {DATA / 'synthetic_perfusion.npz'} ({curves.shape[0]} curves)")


def make_reference_hybrid() -> None:
    prm = Parameters(seed=0)
    grid = Grid(nx=140, ny=160, lx=140, ly=160)
    res = run_hybrid(prm, grid, domain_limit=(-70, 70, 0, 160), eps=4.0)
    tof = res.tof[np.isfinite(res.tof) & (res.tof > 0)]
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    np.savez(
        TEST_DATA / "reference_hybrid.npz",
        n_art_nodes=res.art.n_nodes, n_ven_nodes=res.ven.n_nodes,
        q_a0=res.solution.q_a[0], q_v0=res.solution.q_v[0],
        pressure_min=res.pressure.min(), pressure_max=res.pressure.max(),
        flux_mean=res.flux.mean(), flux_std=res.flux.std(),
        tof_mean=tof.mean(), tof_std=tof.std(),
    )
    print(f"wrote {TEST_DATA / 'reference_hybrid.npz'}")


if __name__ == "__main__":
    make_synthetic_perfusion()
    make_reference_hybrid()
