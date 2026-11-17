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


def geometry_fixed(names, art=None, ven=None, trees=None) -> bool:
    """Whether varying ``names`` leaves the Darcy problem unchanged across samples:
    no permeability change, and the vessel terminals don't move — i.e. tree depth
    (``r_min_frac``) is not varied, or the trees are fixed (given, not rebuilt)."""
    if "permeability" in names:
        return False
    if "r_min_frac" in names:
        return art is not None and ven is not None and trees is None
    return True


def _sample_solver(names, grid, domain_limit, art, ven, trees):
    """``(prm, grid) -> HybridResult`` for a sweep. When the geometry is fixed across
    samples the trees and the Darcy factorisation are built once (lazily, from the
    first sample) and reused — identical results, ~one LU instead of one per sample."""
    from .core.pipeline import darcy_response_for

    if not geometry_fixed(names, art, ven, trees):
        def solve(prm, g):
            a, v = trees(prm) if trees else (art, ven)
            return run_hybrid(prm, g, domain_limit=domain_limit, art=a, ven=v)
        return solve

    cache = {}

    def solve_reusing(prm, g):
        if not cache:
            a, v = trees(prm) if trees else (art, ven)
            cache["a"], cache["v"], cache["r"] = darcy_response_for(prm, g, domain_limit, a, v)
        return run_hybrid(prm, g, domain_limit=domain_limit, art=cache["a"], ven=cache["v"],
                          darcy_response=cache["r"])
    return solve_reusing


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
class SobolResult:
    names: list                  # input names
    qoi_names: list              # output names
    n_samples: int               # LHS samples used to fit the surrogate
    first_order: dict            # qoi -> {input: S1}  (variance fraction from that input alone)
    total_effect: dict           # qoi -> {input: ST}  (incl. all interactions with it)
    fit_r2: dict                 # qoi -> surrogate R² (how well the PCE explains the variance)


# Orthonormal Legendre polynomials on [-1, 1] (unit variance under the uniform measure).
def _legendre(x: np.ndarray, degree: int) -> np.ndarray:
    if degree == 0:
        return np.ones_like(x)
    if degree == 1:
        return np.sqrt(3.0) * x
    return np.sqrt(5.0) * (3.0 * x**2 - 1.0) / 2.0


def _pce_basis(names):
    """Degree-2 polynomial-chaos basis: constant, univariate P1/P2, pairwise P1·P1.

    Returns ``(terms, supports)`` where each term is ``[(input_index, degree), ...]``
    and ``supports`` the set of inputs it depends on (for grouping Sobol indices).
    """
    d = len(names)
    terms, supports = [[]], [frozenset()]                 # constant term
    for i in range(d):
        for p in (1, 2):
            terms.append([(i, p)])
            supports.append(frozenset({i}))
    for i in range(d):                                    # pairwise degree-1 interactions
        for j in range(i + 1, d):
            terms.append([(i, 1), (j, 1)])
            supports.append(frozenset({i, j}))
    return terms, supports


def sobol_indices(param_ranges: dict, grid: Grid, n_samples: int = 48,
                  base: dict | None = None, seed: int = 0,
                  domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
                  art: Network | None = None, ven: Network | None = None,
                  trees=None) -> SobolResult:
    """Variance-based (Sobol) sensitivity via a polynomial-chaos surrogate.

    Fits an orthonormal-Legendre degree-2 expansion to ``n_samples`` Latin-hypercube
    evaluations (the model is smooth and low-dimensional, so this needs far fewer solves
    than raw Saltelli Monte-Carlo and yields stable indices). Sobol indices are then read
    off analytically from the coefficients: ``S1`` is the variance fraction from an
    input's own terms, ``ST`` adds its interaction terms, so ``ST ≥ S1``. ``fit_r2``
    reports how much output variance the surrogate captures (trust the indices when it's
    near 1).
    """
    names = list(param_ranges)
    if not names:
        raise ValueError("param_ranges is empty")
    if n_samples < 6:
        raise ValueError("n_samples must be >= 6 to fit the surrogate")
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n_samples, len(names), rng)        # (N, d) in [0,1)
    lo = np.array([param_ranges[k][0] for k in names], dtype=float)
    hi = np.array([param_ranges[k][1] for k in names], dtype=float)
    samples = lo + unit * (hi - lo)

    solver = _sample_solver(names, grid, domain_limit, art, ven, trees)
    rows = []
    for row in samples:
        kw, perm = dict(base or {}), None
        for name, value in zip(names, row, strict=True):
            if name == "permeability":
                perm = float(value)
            elif name in _PARAM_FIELDS:
                kw[name] = float(value)
            else:
                raise ValueError(f"unknown uncertain parameter {name!r}")
        prm = Parameters(**kw)
        g = grid if perm is None else replace(grid, permeability=perm)
        rows.append(_qois(solver(prm, g)))

    qoi_names = list(rows[0])
    outputs = np.array([[r[q] for q in qoi_names] for r in rows])   # (N, q)
    x = 2.0 * unit - 1.0                                            # map to [-1, 1]
    terms, supports = _pce_basis(names)
    design = np.empty((n_samples, len(terms)))
    for k, term in enumerate(terms):
        col = np.ones(n_samples)
        for i, p in term:
            col = col * _legendre(x[:, i], p)
        design[:, k] = col

    first, total, fit_r2 = {}, {}, {}
    for j, q in enumerate(qoi_names):
        y = outputs[:, j]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        y_var = float(y.var()) or np.nan
        fit_r2[q] = float(np.clip(1.0 - resid.var() / y_var, 0.0, 1.0)) if y_var == y_var else 0.0
        power = coef[1:] ** 2                                       # drop the constant term
        supp = supports[1:]
        var_total = float(power.sum()) or np.nan
        s1, st = {}, {}
        for i, name in enumerate(names):
            s1[name] = float(sum(power[k] for k, s in enumerate(supp) if s == {i}) / var_total)
            st[name] = float(sum(power[k] for k, s in enumerate(supp) if i in s) / var_total)
        first[q], total[q] = s1, st
    return SobolResult(names=names, qoi_names=qoi_names, n_samples=n_samples,
                       first_order=first, total_effect=total, fit_r2=fit_r2)


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
                art: Network | None = None, ven: Network | None = None,
                trees=None) -> UQResult:
    """Monte-Carlo UQ over ``param_ranges`` (``{name: (low, high)}``).

    Each input is either a :class:`Parameters` field or ``"permeability"`` (varied on
    the grid). ``base`` sets the fixed Parameters values; ``art``/``ven`` fix the
    vasculature (else fractal trees are (re)built per sample, so ``r_min_frac``
    uncertainty is only meaningful with fractal trees); ``trees(prm) -> (art, ven)``
    rebuilds them per sample in a custom way (e.g. fitted into a project's domain).
    Returns the QoI distribution and first-order sensitivities.
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
    solver = _sample_solver(names, grid, domain_limit, art, ven, trees)
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
        rows.append(_qois(solver(prm, g)))

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
