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
from .schemas import ExploreGrowRequest, ExploreOptimizeRequest, ExploreRequest

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


def _default_roots(req: ExploreGrowRequest):
    """Default roots: artery at the bottom/base pointing in, vein at the top."""
    from .. import treegen

    if req.dim == 3:
        x0, x1, y0, y1, z0, z1 = treegen.DOMAIN_3D
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return ([cx, cy, z0 + 3], [0.0, 0.0, 1.0]), ([cx, cy, z1 - 3], [0.0, 0.0, -1.0])
    x0, x1, y0, y1 = treegen.DOMAIN_2D
    cx = (x0 + x1) / 2
    return ([cx, y0 + 8], None), ([cx, y1 - 8], None)


def grow_spec(req: ExploreGrowRequest):
    """Translate the request into a :class:`perfusion.treegen.GenSpec`."""
    from .. import treegen

    (a_root, a_dir), (v_root, v_dir) = _default_roots(req)

    def tree(t, root, direction):
        r = t.root if t.root and len(t.root) == req.dim else root
        # an explicitly sent 3-D direction wins; otherwise point into the tissue
        d = t.direction if "direction" in t.model_fields_set else (direction or t.direction)
        if len(d) != 3 or not np.any(d):
            raise ValueError("direction must be a non-zero [dx, dy, dz]")
        return treegen.TreeSpec(root=tuple(r), angle=t.angle,
                                direction=tuple(d), radius=t.radius,
                                gamma=t.gamma, branch_angle=t.branch_angle)

    nx = min(req.nx, 30) if req.dim == 3 else req.nx
    return treegen.GenSpec(
        dim=req.dim, shape=req.shape, polygon=req.polygon, size=req.size,
        artery=tree(req.artery, a_root, a_dir), vein=tree(req.vein, v_root, v_dir),
        generations=req.generations, length_to_radius=req.length_to_radius,
        exponent=req.exponent, fill=req.fill, fill_target=req.fill_target,
        fill_steps=req.fill_steps, n_points=req.n_points, step=req.step,
        influence=req.influence, n_terminals=req.n_terminals, leaf_radius=req.leaf_radius,
        seed=req.seed, nx=nx, ny=req.ny)


def _voxels(pressure, mask, domain, budget=3500):
    """Subsampled active voxels ``[x, y, z, p]`` for the 3-D canvas viewer."""
    lo, hi = np.asarray(domain[0::2]), np.asarray(domain[1::2])
    h = (hi - lo) / np.array(mask.shape)
    k = max(1, round((mask.sum() / budget) ** (1 / 3)))           # stride
    sub = mask[::k, ::k, ::k] & np.isfinite(pressure[::k, ::k, ::k])
    idx = np.argwhere(sub) * k
    xyz = np.round(lo + (idx + 0.5) * h, 1)
    p = np.round(pressure[tuple(idx.T)].astype(float), 5)
    return np.column_stack([xyz, p]).tolist()


@router.post("/explore/grow")
def explore_grow(req: ExploreGrowRequest) -> dict:
    """Grow artery + vein into a tissue shape (2-D or 3-D) and optionally solve.

    Returns the tissue (2-D outline polygon, or 3-D shape + domain), the roots,
    both trees' segments and stats, and — when ``solve`` — the field (2-D heatmap,
    or 3-D pressure voxels), inflow and the homogeneity objective (2-D)."""
    from .. import treegen

    try:
        spec = grow_spec(req)
        art, ven, mask, domain, outline = treegen.build(spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None

    if req.dim == 3:
        def segs(net, k):
            return [[*map(lambda v: round(v, 2), s[:6]), round(s[6], 3), k]
                    for s in helpers.segment_lines_3d(net)]
    else:
        def segs(net, k):
            return [[x0, y0, x1, y1, round(r, 3), k]
                    for x0, y0, x1, y1, r in helpers.segment_lines(net)]

    out = {
        "dim": req.dim, "domain": list(domain), "outline": outline, "shape": req.shape,
        "roots": {"artery": list(spec.artery.root), "vein": list(spec.vein.root)},
        "segments": segs(art, 0) + segs(ven, 1),
        "stats": {"artery": treegen.tree_stats(art), "vein": treegen.tree_stats(ven)},
        "active_cells": int(mask.sum()),
    }
    if req.dim == 3:          # a light point cloud of the tissue, drawn when unsolved
        out["tissue_points"] = [v[:3] for v in _voxels(np.zeros(mask.shape), mask, domain,
                                                       budget=1200)]
    if not req.solve:
        return out
    try:
        res = treegen.solve(spec, art, ven, mask, domain)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        out["solve_error"] = f"this configuration did not solve: {exc}"
        return out
    q = res.solution.terminal_flux
    out["inflow"] = float(q[q > 0].sum())
    if req.dim == 3:
        p = np.where(mask, res.pressure, np.nan)
        fin = p[np.isfinite(p)]
        out["field"] = {"name": "pressure", "min": float(fin.min()), "max": float(fin.max()),
                        "voxels": _voxels(res.pressure, mask, domain)}
        return out
    if req.field not in ("pressure", "flux", "tof"):
        raise HTTPException(422, "field must be 'pressure', 'flux' or 'tof'")
    fld = {"pressure": res.pressure, "flux": res.flux, "tof": res.tof}[req.field]
    fld = np.where(mask, fld, np.nan)
    fin = fld[np.isfinite(fld)]
    out["field"] = {"name": req.field, "shape": list(fld.shape),
                    "vmin": float(fin.min()) if fin.size else 0.0,
                    "vmax": float(fin.max()) if fin.size else 1.0,
                    "data": np.where(np.isfinite(fld), fld, None).tolist()}
    out["objective"] = _objective_of(res, res.grid, req.kind)
    out["kind"] = req.kind
    return out
