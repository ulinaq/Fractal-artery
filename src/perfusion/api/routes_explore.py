"""Tree Explorer routes — project-independent fractal-tree exploration.

Unlike the project workflow (arbitrary drawn/segmented networks), the explorer
works in the thesis' native frame: it builds a fractal tree from a handful of
``optimize.alpha`` tunables, solves the hybrid model on a coarse grid, and
returns the field + tree geometry + a homogeneity objective — so the Paper-D
optimisation is reachable interactively without shoehorning it into a project.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from .. import optimize
from ..core.darcy import Grid
from ..core.pipeline import run_hybrid
from . import helpers
from .schemas import ExploreOptimizeRequest, ExploreRequest

router = APIRouter()

_DOMAIN = (-70.0, 70.0, 0.0, 160.0)
_RMIN = 0.4          # shallow tree -> fast, interactive solves


def _alpha_from(req: ExploreRequest):
    a = optimize.default_alpha()
    a[0], a[1], a[2] = req.a_gamma, req.v_gamma, req.length_to_radius
    a[3], a[8] = req.rv_scale, req.exponent
    return a


def _objective_of(res, grid, kind: str) -> float:
    mask = grid.mask
    if kind.startswith("flux"):
        v = res.flux[mask]
        return {"flux_range": float(v.max() - v.min()), "flux_std": float(v.std()),
                "flux_l2": float(np.linalg.norm(v)), "flux_mean": float(v.mean())}[kind]
    tof = res.tof[mask]
    tof = tof[np.isfinite(tof) & (tof > 0)]
    if tof.size == 0:
        return float("inf")
    return float(tof.std()) if kind == "tof_std" else float(tof.max())


@router.post("/explore")
def explore(req: ExploreRequest) -> dict:
    """Solve one fractal-tree configuration; return field + tree + objective."""
    if req.field not in ("pressure", "flux", "tof"):
        raise HTTPException(422, "field must be 'pressure', 'flux' or 'tof'")
    grid = Grid(nx=req.nx, ny=req.ny, lx=140.0, ly=160.0)
    try:
        res = run_hybrid(optimize.params_from_alpha(_alpha_from(req), r_min_frac=_RMIN),
                         grid, _DOMAIN)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(422, f"this configuration did not solve: {exc}") from None

    fld = {"pressure": res.pressure, "flux": res.flux, "tof": res.tof}[req.field]
    fld = np.where(grid.mask, fld, np.nan)
    finite = fld[np.isfinite(fld)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    data = np.where(np.isfinite(fld), fld, None)

    def segs(net, kind):
        return [[x0, y0, x1, y1, round(r, 3), kind]
                for x0, y0, x1, y1, r in helpers.segment_lines(net)]

    return {
        "objective": _objective_of(res, grid, req.kind), "kind": req.kind,
        "domain": list(_DOMAIN),
        "field": {"name": req.field, "shape": list(fld.shape),
                  "vmin": vmin, "vmax": vmax, "data": data.tolist()},
        "segments": segs(res.art, 0) + segs(res.ven, 1),
    }


@router.post("/explore/optimize")
def explore_optimize(req: ExploreOptimizeRequest) -> dict:
    """Optimise the tree for the chosen homogeneity objective; return the best
    tunables and the before/after objective (bounded budget — runs inline)."""
    grid = Grid(nx=req.nx, ny=req.ny, lx=140.0, ly=160.0)
    start = optimize.objective(optimize.default_alpha(), kind=req.kind, grid=grid,
                               domain_limit=_DOMAIN, r_min_frac=_RMIN)
    try:
        r = optimize.optimize_tree(kind=req.kind, grid=grid, domain_limit=_DOMAIN,
                                   method=req.method, maxiter=req.maxiter, r_min_frac=_RMIN)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    a = r.alpha
    improvement = (start - r.objective) / start if np.isfinite(start) and start > 0 else 0.0
    return {
        "kind": r.kind, "start": start, "best": r.objective,
        "improvement": improvement, "n_eval": r.n_eval, "success": r.success,
        "a_gamma": a[0], "v_gamma": a[1], "length_to_radius": a[2],
        "rv_scale": a[3], "exponent": a[8],
    }
