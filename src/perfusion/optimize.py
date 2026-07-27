"""Vascular-structure optimisation objectives (Paper D).

Optimise the fractal-tree configuration so blood is delivered to the whole
capillary bed as evenly as possible. Two families of objective are provided:
flux-homogeneity (range / std / L2 / mean) and time-of-flight homogeneity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core.darcy import Grid
from .core.parameters import Parameters, bifurcation_geometry
from .core.pipeline import run_hybrid

# Parameter-vector layout (mirrors Tree_parameter.m ``alpha``), 0-based:
# 0: a_gamma        1: v_gamma       2: length_to_radius   3: rv_scale
# 4: a_theta        5: v_theta       6: a_init_x           7: v_init_x
# 8: exponent       9-12: deeper arterial gammas   13-19: deeper venous gammas
ALPHA_LEN = 20


def default_alpha() -> np.ndarray:
    a = np.zeros(ALPHA_LEN)
    a[0], a[1] = 1.4, 1.2
    a[2] = 10.0
    a[3] = 1.2
    a[4] = a[5] = np.pi / 2
    a[6], a[7] = 35.0, 30.0
    a[8] = 2.8
    a[9:] = 1.0
    return a


def params_from_alpha(alpha: Sequence[float]) -> Parameters:
    """Build :class:`Parameters` from an optimisation vector."""
    a = np.asarray(alpha, dtype=float)
    prm = Parameters(
        length_to_radius=a[2], exponent=a[8], rv_scale=a[3],
        a_init=(a[6], 0.0), v_init=(a[7], 160.0),
        a_theta=a[4], v_theta=a[5], a_gamma=a[0], v_gamma=a[1],
    )
    # Override the deeper-generation asymmetries deterministically from alpha.
    p0, r0 = bifurcation_geometry(a[0], a[8])
    phis, ratios = [p0], [r0]
    for g in a[9:13]:
        p, r = bifurcation_geometry(g, a[8])
        phis.append(p)
        ratios.append(r)
    prm.phi = np.array(phis)
    prm.ratio = np.array(ratios)
    vphis = [bifurcation_geometry(a[1], a[8])[0]]
    vratios = [bifurcation_geometry(a[1], a[8])[1]]
    for g in a[13:20]:
        p, r = bifurcation_geometry(g, a[8])
        vphis.append(p)
        vratios.append(r)
    prm.vphi = np.array(vphis)
    prm.vratio = np.array(vratios)
    return prm


def objective(alpha: Sequence[float], kind: str = "tof_std",
              grid: Grid | None = None,
              domain_limit=(-70, 70, 0, 160), strict: bool = False) -> float:
    """Scalar objective for an optimisation vector.

    ``kind`` in {``flux_range``, ``flux_std``, ``flux_l2``, ``flux_mean``,
    ``tof_std``, ``tof_max``}. With ``strict=True`` a tree whose terminals leave
    the grid returns ``inf`` (so an optimiser hard-rejects it); by default the
    terminals are clipped to the grid, keeping the objective smooth.
    """
    if grid is None:
        grid = Grid(nx=140, ny=160, lx=140, ly=160)
    try:
        res = run_hybrid(params_from_alpha(alpha), grid, domain_limit,
                         strict_domain=strict)
    except (ValueError, np.linalg.LinAlgError):
        return float("inf")

    mask = grid.mask
    if kind.startswith("flux"):
        v = res.flux[mask]
        return {
            "flux_range": float(v.max() - v.min()),
            "flux_std": float(v.std()),
            "flux_l2": float(np.linalg.norm(v)),
            "flux_mean": float(v.mean()),
        }[kind]
    tof = res.tof[mask]
    tof = tof[np.isfinite(tof) & (tof > 0)]
    if tof.size == 0:
        return float("inf")
    if kind == "tof_std":
        return float(tof.std())
    if kind == "tof_max":
        return float(tof.max())
    raise ValueError(f"unknown objective {kind!r}")


# ── optimisation driver (Paper D) ──────────────────────────────────────────────
# Which alpha components to vary by default, and sensible bounds for each. The
# defaults vary just the two branching asymmetries (a_gamma, v_gamma) — the main
# levers on delivery homogeneity — keeping the search cheap and well-behaved.
_DEFAULT_FREE = (0, 1)
_ALPHA_BOUNDS = {
    0: (1.0, 1.8), 1: (1.0, 1.8),        # arterial / venous gamma (asymmetry)
    2: (5.0, 16.0),                       # length-to-radius
    3: (0.8, 1.6),                        # venous radius scale
    8: (2.2, 3.4),                        # bifurcation exponent
}


@dataclass
class OptimizeResult:
    """Outcome of :func:`optimize_tree`."""

    alpha: list[float]      # full best parameter vector (length ALPHA_LEN)
    params: Parameters      # the corresponding Parameters
    objective: float        # best objective value
    kind: str               # objective minimised
    free: list[int]         # alpha components that were varied
    n_eval: int             # number of objective evaluations (hybrid solves)
    success: bool


def optimize_tree(kind: str = "tof_std", *, grid: Grid | None = None,
                  domain_limit=(-70, 70, 0, 160), free: Sequence[int] | None = None,
                  bounds: Sequence[tuple[float, float]] | None = None,
                  method: str = "local", maxiter: int = 40, seed: int = 0,
                  strict: bool = False, alpha0: Sequence[float] | None = None) -> OptimizeResult:
    """Optimise the fractal-tree configuration to homogenise perfusion (Paper D).

    Minimises :func:`objective` (``kind`` selects flux- or TOF-homogeneity) over
    the ``free`` components of the ``alpha`` vector; the rest stay at ``alpha0``
    (default :func:`default_alpha`). ``method`` is ``"local"`` (Nelder-Mead from
    ``alpha0``) or ``"global"`` (differential evolution — a population search that
    tolerates the objective's flat/clipped regions). Each evaluation is a full
    hybrid solve, so keep the grid coarse and ``maxiter`` modest.
    """
    from scipy.optimize import differential_evolution, minimize

    grid = grid or Grid(nx=70, ny=80, lx=140, ly=160)
    free = list(_DEFAULT_FREE if free is None else free)
    base = np.array(default_alpha() if alpha0 is None else alpha0, dtype=float)
    bnds = list(bounds) if bounds is not None else [_ALPHA_BOUNDS.get(i, (0.5, 3.5)) for i in free]

    n_eval = 0

    def f(x):
        nonlocal n_eval
        n_eval += 1
        a = base.copy()
        a[free] = x
        return objective(a, kind=kind, grid=grid, domain_limit=domain_limit, strict=strict)

    if method == "global":
        res = differential_evolution(f, bnds, maxiter=maxiter, seed=seed,
                                     popsize=8, tol=1e-3, polish=False)
        x_best, f_best, ok = res.x, float(res.fun), bool(res.success)
    elif method == "local":
        res = minimize(f, base[free], method="Nelder-Mead", bounds=bnds,
                       options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4})
        x_best, f_best, ok = res.x, float(res.fun), bool(res.success)
    else:
        raise ValueError(f"unknown method {method!r}; use 'local' or 'global'")

    best = base.copy()
    best[free] = x_best
    return OptimizeResult(alpha=best.tolist(), params=params_from_alpha(best),
                          objective=f_best, kind=kind, free=free, n_eval=n_eval, success=ok)
