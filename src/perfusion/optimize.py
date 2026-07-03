"""Vascular-structure optimisation objectives (Paper D).

Optimise the fractal-tree configuration so blood is delivered to the whole
capillary bed as evenly as possible. Two families of objective are provided:
flux-homogeneity (range / std / L2 / mean) and time-of-flight homogeneity.
"""

from __future__ import annotations

from collections.abc import Sequence

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
