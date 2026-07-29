"""Uncertainty quantification for the perfusion model (Monte-Carlo + sensitivity).

Sample uncertain inputs over given ranges with Latin-hypercube sampling, run the
hybrid model at each sample, and report the distribution of quantities of interest
(arterial inflow, mean Darcy flux, mean transit time) together with a first-order
**sensitivity** — the Pearson correlation between each input and each output, a cheap
ranking of which parameter drives which response.

A service layer built on the pure ``core`` engine (like ``optimize``): no I/O, viz
or web dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .core.darcy import Grid
from .core.geometry import Network
from .core.parameters import Parameters
from .core.pipeline import run_hybrid

# Inputs that map onto a Parameters field; "permeability" is handled on the Grid.
_PARAM_FIELDS = ("p_in", "p_out", "mu", "r_min_frac", "hematocrit")


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube samples in the unit cube ``[0, 1)^d`` (stratified per axis)."""
    edges = np.arange(n)[:, None]
    jitter = rng.uniform(size=(n, d))
    samples = (edges + jitter) / n
    for j in range(d):                       # independent per-dimension permutation
        rng.shuffle(samples[:, j])
    return samples


def _qois(result) -> dict:
    """Scalar quantities of interest extracted from a hybrid solve."""
    tf = result.solution.terminal_flux
    flux = result.flux[np.isfinite(result.flux)]
    tof = result.tof[np.isfinite(result.tof)]
    return {
        "arterial_inflow": float(tf[tf > 0].sum()),
        "flux_mean": float(flux.mean()) if flux.size else 0.0,
        "tof_mean": float(tof.mean()) if tof.size else 0.0,
    }


@dataclass
class UQResult:
    names: list                  # uncertain input names (columns of ``samples``)
    qoi_names: list              # output names (columns of ``outputs``)
    samples: np.ndarray          # (n, d) sampled inputs
    outputs: np.ndarray          # (n, q) QoI values
    mean: dict                   # qoi -> mean
    std: dict                    # qoi -> standard deviation
    p05: dict                    # qoi -> 5th percentile
    p95: dict                    # qoi -> 95th percentile
    sensitivity: dict            # qoi -> {input: Pearson r in [-1, 1]}


def monte_carlo(param_ranges: dict, grid: Grid, n_samples: int = 32,
                base: dict | None = None, seed: int = 0,
                domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
                art: Network | None = None, ven: Network | None = None) -> UQResult:
    """Monte-Carlo UQ over ``param_ranges`` (``{name: (low, high)}``).

    Each input is either a :class:`Parameters` field or ``"permeability"`` (varied on
    the grid). ``base`` sets the fixed Parameters values; ``art``/``ven`` fix the
    vasculature (else fractal trees are (re)built per sample, so ``r_min_frac``
    uncertainty is only meaningful with fractal trees). Returns the QoI distribution
    and first-order sensitivities.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for statistics")
    names = list(param_ranges)
    if not names:
        raise ValueError("param_ranges is empty")
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n_samples, len(names), rng)
    lo = np.array([param_ranges[k][0] for k in names], dtype=float)
    hi = np.array([param_ranges[k][1] for k in names], dtype=float)
    samples = lo + unit * (hi - lo)

    base = dict(base or {})
    rows = []
    for row in samples:
        kw, perm = dict(base), None
        for name, value in zip(names, row, strict=True):
            if name == "permeability":
                perm = float(value)
            elif name in _PARAM_FIELDS:
                kw[name] = float(value)
            else:
                raise ValueError(f"unknown uncertain parameter {name!r}")
        prm = Parameters(**kw)
        g = grid if perm is None else replace(grid, permeability=perm)
        rows.append(_qois(run_hybrid(prm, g, domain_limit=domain_limit, art=art, ven=ven)))

    qoi_names = list(rows[0])
    outputs = np.array([[r[q] for q in qoi_names] for r in rows])

    def stat(fn):
        return {q: float(fn(outputs[:, i])) for i, q in enumerate(qoi_names)}

    sensitivity = {}
    for i, q in enumerate(qoi_names):
        y = outputs[:, i]
        row = {}
        for j, name in enumerate(names):
            x = samples[:, j]
            if x.std() < 1e-12 or y.std() < 1e-12:
                row[name] = 0.0
            else:
                row[name] = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
        sensitivity[q] = row

    return UQResult(
        names=names, qoi_names=qoi_names, samples=samples, outputs=outputs,
        mean=stat(np.mean), std=stat(np.std),
        p05=stat(lambda a: np.percentile(a, 5)), p95=stat(lambda a: np.percentile(a, 95)),
        sensitivity=sensitivity,
    )
