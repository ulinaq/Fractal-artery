"""Tree Explorer routes — project-independent fractal-tree exploration.

Unlike the project workflow (arbitrary drawn/segmented networks), the explorer
works in the thesis' native frame: it builds a fractal tree from a handful of
``optimize.alpha`` tunables, solves the hybrid model on a coarse grid, and
returns the field + tree geometry + a homogeneity objective — so the Paper-D
optimisation is reachable interactively without shoehorning it into a project.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .. import optimize
from ..core.darcy import Grid
from ..core.pipeline import run_hybrid
from . import helpers
from .deps import get_store
from .schemas import (
    ExploreGrowRequest,
    ExploreOptimizeRequest,
    ExploreProjectRequest,
    ExploreRequest,
    ShapePolygonRequest,
)

router = APIRouter()

_DOMAIN = (-70.0, 70.0, 0.0, 160.0)
_RMIN = 0.4          # shallow tree -> fast, interactive solves


def _alpha_from(req: ExploreRequest):
    a = optimize.default_alpha()
    a[0], a[1], a[2] = req.a_gamma, req.v_gamma, req.length_to_radius
    a[3], a[8] = req.rv_scale, req.exponent
    return a


def _objective_of(res, grid, kind: str) -> float:
    """The chosen homogeneity objective of a 2-D or 3-D result."""
    from ..treegen import homogeneity

    metrics = homogeneity(res)
    if kind not in metrics:
        raise HTTPException(422, f"kind must be one of {sorted(metrics)}")
    return metrics[kind]


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


def _default_roots(req: ExploreGrowRequest) -> dict:
    """Where the organ's artery and vein attach anatomically (``perfusion.organs``):
    e.g. the renal hilum, the aortic root of the heart, the porta hepatis. A custom
    2-D outline falls back to bottom-in / top-out."""
    from .. import organs, treegen

    dom = treegen.domain_for(req.dim, req.extent)
    name = req.shape if req.shape != "custom" else "ellipse"
    return organs.roots(name, req.dim, dom, req.size)


def grow_spec(req: ExploreGrowRequest):
    """Translate the request into a :class:`perfusion.treegen.GenSpec`."""
    from .. import treegen

    dflt = _default_roots(req)

    def tree(t, kind):
        d0 = dflt[kind]
        given = bool(t.root) and len(t.root) == req.dim
        root = t.root if given else d0["root"]
        # an aim the client sent wins; otherwise the organ's anatomical one
        angle = t.angle if given and "angle" in t.model_fields_set else d0.get("angle", t.angle)
        if req.dim == 3:
            d = (t.direction if given and "direction" in t.model_fields_set
                 else d0["direction"])
            if len(d) != 3 or not np.any(d):
                raise ValueError("direction must be a non-zero [dx, dy, dz]")
        else:
            d = t.direction
        return treegen.TreeSpec(root=tuple(root), angle=angle, direction=tuple(d),
                                radius=t.radius, gamma=t.gamma)

    nx = min(req.nx, 36) if req.dim == 3 else req.nx
    return treegen.GenSpec(
        dim=req.dim, shape=req.shape, polygon=req.polygon, size=req.size,
        extent=req.extent, twist=req.twist, jitter=req.jitter,
        artery=tree(req.artery, "artery"), vein=tree(req.vein, "vein"),
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
    both trees' segments and stats, and — when ``solve`` — the chosen field
    (pressure / flux / time-of-flight: a 2-D heatmap or 3-D voxels), the inflow and
    the homogeneity objective, computed the same way in 2-D and 3-D."""
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
        # the trunk aims used (2-D angle in degrees / 3-D direction)
        "aims": {k: (list(t.direction) if req.dim == 3 else t.angle)
                 for k, t in (("artery", spec.artery), ("vein", spec.vein))},
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
    if req.field not in ("pressure", "flux", "tof"):
        raise HTTPException(422, "field must be 'pressure', 'flux' or 'tof'")
    fld = {"pressure": res.pressure, "flux": res.flux, "tof": res.tof}[req.field]
    fld = np.where(mask, fld, np.nan)
    fin = fld[np.isfinite(fld)]
    if req.dim == 3:
        # robust colour range: time-of-flight is heavy-tailed (stagnant corners)
        lo, hi = (np.percentile(fin, [2, 98]) if fin.size else (0.0, 1.0))
        out["field"] = {"name": req.field, "min": float(lo), "max": float(hi),
                        "voxels": _voxels(fld, mask, domain)}
    else:
        out["field"] = {"name": req.field, "shape": list(fld.shape),
                        "vmin": float(fin.min()) if fin.size else 0.0,
                        "vmax": float(fin.max()) if fin.size else 1.0,
                        "data": np.where(np.isfinite(fld), fld, None).tolist()}
    out["objective"] = _objective_of(res, res.grid, req.kind)
    out["kind"] = req.kind
    return out


@router.get("/shapes")
def shapes() -> dict:
    """The organ / tissue shapes available in 2-D and 3-D (name → label)."""
    from .. import organs

    return {"2d": organs.SHAPES_2D, "3d": organs.SHAPES_3D}


@router.post("/shapes/polygon")
def shape_polygon(req: ShapePolygonRequest) -> dict:
    """An organ outline placed in a 2-D window ``[x0, x1, y0, y1]`` — e.g. to drop a
    kidney/heart/liver tissue outline into a workspace project."""
    from .. import organs

    if len(req.domain) != 4 or req.domain[1] <= req.domain[0] or req.domain[3] <= req.domain[2]:
        raise HTTPException(422, "domain must be [x0, x1, y0, y1] with x1 > x0, y1 > y0")
    try:
        return {"name": req.name,
                "polygon": organs.polygon(req.name, req.domain, req.size, seed=req.seed)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.post("/explore/project")
def explore_to_project(req: ExploreProjectRequest, store=Depends(get_store)) -> dict:
    """Turn the current explorer setup into a **project**.

    2-D → a workspace project: the organ outline as an editable tissue region and
    the grown artery/vein as editable networks. 3-D → a 3-D project (``grid.nz``,
    raster tissue mask, 3-D trees) to run and view in the 3-D viewer. Returns the
    new project's id and the page to open."""
    from .. import treegen
    from .schemas import GridIn, ParametersIn

    try:
        spec = grow_spec(req)
        art, ven, mask, domain, outline = treegen.build(spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    dom = [float(v) for v in domain]
    if req.dim == 3:
        n = mask.shape[0]
        grid = GridIn(nx=n, ny=n, nz=n, lx=dom[1] - dom[0], ly=dom[3] - dom[2],
                      lz=dom[5] - dom[4], domain_limit=tuple(dom), tissue=req.shape)
    else:
        grid = GridIn(nx=spec.nx, ny=spec.ny, lx=dom[1] - dom[0], ly=dom[3] - dom[2],
                      domain_limit=tuple(dom))
    project = store.create(req.name, ParametersIn(), grid)
    project.artery_network, project.vein_network = art, ven
    if req.dim == 3:
        project.mask = np.asarray(mask, dtype=bool)
    else:
        project.regions = [outline]
    store.save(project)
    page = (f"/viewer3d.html?project={project.id}" if req.dim == 3
            else f"/?project={project.id}")
    return {"id": project.id, "dim": req.dim, "open": page}
