"""Model building + job execution, independent of the web app.

Kept out of ``app.py`` so both the FastAPI endpoints and background workers (the
thread pool *and* an optional RQ worker process) can build the model and run a
project without importing the app. Functions take a ``ProjectStore`` + a
``Project`` and persist their results through the store.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .. import tkm, transport
from ..core.geometry import (
    arterial_tree,
    network_from_polylines,
    rasterize_mask,
    venous_tree,
)
from ..core.pipeline import run_hybrid
from .schemas import RunRequest, TracerRequest


def polylines_to_network(polylines, root_xy=None):
    return network_from_polylines([p.points for p in polylines],
                                  radii=[p.radius for p in polylines], root_xy=root_xy)


def network_from_strokes(polylines, root_xy=None):
    """Build a network from polylines while keeping each stroke's *shape*.

    Unlike :func:`polylines_to_network` (which collapses a stroke to a single
    straight edge between its endpoints — turning a bent vessel into a chord),
    this explodes every stroke into consecutive 2-point edges, so bends stay as
    degree-2 nodes and each edge carries the stroke's radius. Used as the base
    for growth so grown branches extend the drawn vessel, not a straight line.
    """
    from itertools import pairwise

    segs, radii = [], []
    for p in polylines:
        pts = [list(pt) for pt in p.points]
        for a, b in pairwise(pts):
            segs.append([a, b])
            radii.append(p.radius)
    if not segs:
        return None
    return network_from_polylines(segs, radii=radii, root_xy=root_xy, snap_tol=0.5)


def is_3d(project) -> bool:
    return bool(getattr(project.grid, "is_3d", False))


def region_field(regions, key: str, shape, domain_limit, base: float) -> np.ndarray:
    """Rasterise stored lesions / hotspots (``key`` = their value field) — polygons,
    discs or spheres — onto a 2-D or 3-D grid (see ``geometry.rasterize_regions``)."""
    from ..core.geometry import rasterize_regions

    return rasterize_regions(shape, domain_limit,
                             [{**r, "value": r[key]} for r in regions], base=base)


def build_grid3d(project):
    """A 3-D project's ``Grid3D`` (+ 6-value window) with its raster tissue mask and,
    with lesions, the heterogeneous per-voxel permeability (spheres of low K)."""
    g = project.grid
    raster = getattr(project, "mask", None)
    mask = None
    if raster is not None and np.asarray(raster).shape == (g.nx, g.ny, g.nz):
        mask = np.asarray(raster, dtype=bool)
    grid = g.to_core3d(mask=mask)
    lesions = getattr(project, "lesions", None)
    if lesions:
        grid.permeability = region_field(lesions, "permeability", (g.nx, g.ny, g.nz),
                                         g.domain_limit, base=g.permeability)
    return grid, tuple(g.domain_limit)


def default_trees_3d(project):
    """Fallback 3-D artery + vein for a 3-D project: fractal trees grown from the
    tissue's anatomical roots (``organs.roots``), sized to the domain — the same
    proportions as the Tree explorer's defaults."""
    from .. import organs
    from ..core.geometry import grow_tree_3d

    dom = tuple(project.grid.domain_limit)
    name = project.grid.tissue if project.grid.tissue in organs.SHAPES_3D else "box"
    roots = organs.roots(name, 3, dom)
    size = float(min(dom[1] - dom[0], dom[3] - dom[2], dom[5] - dom[4]))
    r0 = 0.025 * size
    kw = dict(min_radius=0.0, length_to_radius=5.0, max_generations=4, exponent=2.8,
              jitter=0.3)
    art = grow_tree_3d(roots["artery"]["root"], roots["artery"]["direction"],
                       root_radius=r0, gamma=1.4, seed=0, **kw)
    ven = grow_tree_3d(roots["vein"]["root"], roots["vein"]["direction"],
                       root_radius=1.2 * r0, gamma=1.2, seed=7, mirror=True, **kw)
    return art, ven


def defaults_for(project):
    """The project's fallback (art, ven): 3-D defaults for a 3-D project, else the
    domain-fitted 2-D fractal trees."""
    return default_trees_3d(project) if is_3d(project) else default_trees(project)


def require_2d(project, feature: str) -> None:
    """Raise :class:`InputError` for a 2-D-only feature on a 3-D project."""
    if is_3d(project):
        raise InputError(f"{feature} works on 2-D projects; this is a 3-D project — "
                         "view and run it in the 3-D viewer.")


def build_grid(project):
    """Grid + its domain window, with the tissue mask: a stored raster mask if
    present (shape must match the grid), else the drawn tissue polygons, else none.
    A 3-D project gets its ``Grid3D`` (see :func:`build_grid3d`)."""
    if is_3d(project):
        return build_grid3d(project)
    nx, ny, domain_limit = project.grid.nx, project.grid.ny, project.grid.domain_limit
    raster = getattr(project, "mask", None)
    if raster is not None and np.asarray(raster).shape == (nx, ny):
        mask = np.asarray(raster, dtype=bool)
    elif project.regions:
        mask = rasterize_mask(nx, ny, domain_limit, project.regions)
    else:
        mask = None
    lesions = getattr(project, "lesions", None)
    if lesions:
        kfield = region_field(lesions, "permeability", (nx, ny), domain_limit,
                              base=project.grid.permeability)
        return project.grid.to_core(mask=mask, permeability=kfield), domain_limit
    return project.grid.to_core(mask=mask), domain_limit


def fit_to_domain(nets, domain_limit, margin=0.05):
    """Fit trees (fitted *jointly*, so artery and vein keep their relative layout)
    into ``domain_limit``: one uniform scale — the trees keep their shape and calibre
    ratios — and centred, their combined bounding box filling the window minus a
    ``margin`` fraction."""
    from ..core.geometry import transform_network

    x0, x1, y0, y1 = domain_limit
    xs = np.concatenate([n.node_x for n in nets])
    ys = np.concatenate([n.node_y for n in nets])
    bw, bh = max(np.ptp(xs), 1e-9), max(np.ptp(ys), 1e-9)
    s = (1 - 2 * margin) * min((x1 - x0) / bw, (y1 - y0) / bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bcx, bcy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    return [transform_network(n, s, (cx - s * bcx, cy - s * bcy)) for n in nets]


def default_trees(project, prm=None):
    """The fallback fractal artery + vein for a project with no vessels of that kind,
    fitted into the project's own domain (the raw thesis trees are built in a fixed
    frame and extend partly outside most project domains). ``prm`` lets a sweep
    rebuild them per sample (e.g. varying ``r_min_frac``)."""
    prm = prm if prm is not None else project.params.to_core()
    art, ven = fit_to_domain([arterial_tree(prm), venous_tree(prm)],
                             project.grid.domain_limit)
    return art, ven


def build_networks(project, use_vessels, defaults=False):
    """Arterial/venous networks: a grown/segmented tree wins, else hand-drawn; with
    ``defaults`` a missing tree is the domain-fitted default fractal tree."""
    art = ven = None
    if use_vessels and project.has_vessels:
        art = polylines_to_network(project.arteries) if project.arteries else None
        ven = polylines_to_network(project.veins) if project.veins else None
    if project.artery_network is not None:
        art = project.artery_network
    if project.vein_network is not None:
        ven = project.vein_network
    if defaults and (art is None or ven is None):
        d_art, d_ven = defaults_for(project)
        art = art if art is not None else d_art
        ven = ven if ven is not None else d_ven
    return art, ven


def tree_builder(project):
    """For sweeps (UQ / calibration): ``prm -> (art, ven)`` rebuilding the default
    trees per sample when the project has none (so ``r_min_frac`` matters), else
    ``None`` (the project's own trees are used as-is). A 3-D project's trees don't
    depend on the parameters, so it always returns ``None``."""
    if is_3d(project):
        return None
    art, ven = build_networks(project, use_vessels=True)
    if art is not None and ven is not None:
        return None

    def build(prm):
        d_art, d_ven = default_trees(project, prm)
        return (art if art is not None else d_art), (ven if ven is not None else d_ven)
    return build


def snap_io(net, points):
    """Snap inlet/outlet points to nearest vessel ends; keep every component fed.

    Each connected component (separate feeding tree) must keep >=1 prescribed
    node or its block is singular, so any component the user left unmarked is
    topped up with its own root.
    """
    if net is None or not points:
        return None
    ends = np.unique(np.concatenate([np.flatnonzero(net.node_degree == 1),
                                     net.inlet_nodes]))
    ex, ey = net.node_x[ends], net.node_y[ends]
    idx = {int(ends[int(np.argmin((ex - x) ** 2 + (ey - y) ** 2))]) for x, y in points}
    labels = net.node_components()
    covered = {int(labels[i]) for i in idx}
    for root in net.inlet_nodes:
        if int(labels[root]) not in covered:
            idx.add(int(root))
            covered.add(int(labels[root]))
    return np.array(sorted(idx), dtype=int)


class InputError(ValueError):
    """A run cannot proceed because of a clear, user-fixable input problem.

    Its message is meant for the user (surfaced verbatim in the UI), so keep it
    plain and actionable — don't wrap it in a traceback.
    """


def validate_run(project, use_vessels):
    """Check the model can be solved; raise :class:`InputError` with a clear,
    actionable message for the common misconfigurations. Returns (grid, art, ven)."""
    prm = project.params.to_core()
    if is_3d(project):
        return _validate_run_3d(project, prm)
    grid, _ = build_grid(project)
    if grid.mask is not None:
        active = int(np.asarray(grid.mask).sum())
        if active == 0:
            raise InputError(
                "The tissue mask has no active cells — the organ outline is empty or "
                "falls outside the grid. Redraw the tissue in Tissue mode, or clear it "
                "to use the whole domain.")
        from scipy.ndimage import label
        _, n_blobs = label(np.asarray(grid.mask))
        if n_blobs > 1:
            raise InputError(
                f"The tissue mask is split into {n_blobs} disconnected regions. The Darcy "
                "solve pins one cell, so only the region containing it is determined and the "
                "rest make the system singular. Edit the tissue outline so it is one "
                "connected area (or clear it to use the whole domain).")

    # a missing tree is the default fractal tree, fitted into this project's domain
    art, ven = build_networks(project, use_vessels, defaults=True)
    if prm.p_in == prm.p_out:
        raise InputError(
            f"p_in equals p_out ({prm.p_in}); there is no pressure drop to drive flow. "
            "Set p_in above p_out in Parameters.")
    for net, points, name, word in ((art, project.inlets, "arterial", "inlet"),
                                    (ven, project.outlets, "venous", "outlet")):
        if net is None:
            continue
        snapped = snap_io(net, points)
        eff = net if snapped is None else replace(net, inlet_nodes=snapped)
        if eff.terminal_nodes.size == 0:
            raise InputError(
                f"The {name} tree has no free ends: every end is marked as an {word}, so "
                "it cannot exchange blood with the tissue and no flow can pass. In the "
                f"Vessels tab, select an end node and unset {word} (I) so it couples to "
                "the capillary bed.")
    return grid, art, ven


def _validate_run_3d(project, prm):
    """The 3-D counterpart of :func:`validate_run` → ``(Grid3D, art, ven)``."""
    from scipy.ndimage import label

    grid, _ = build_grid3d(project)
    active = int(grid.mask.sum())
    if active == 0:
        raise InputError("The 3-D tissue mask has no active cells.")
    n_blobs = label(grid.mask)[1]
    if n_blobs > 1:
        raise InputError(f"The 3-D tissue is split into {n_blobs} disconnected regions; "
                         "the Darcy solve needs one connected tissue.")
    if prm.p_in == prm.p_out:
        raise InputError(f"p_in equals p_out ({prm.p_in}); there is no pressure drop to "
                         "drive flow. Set p_in above p_out.")
    art, ven = build_networks(project, use_vessels=True, defaults=True)
    if not (art.is_3d and ven.is_3d):
        raise InputError("A 3-D project needs 3-D vessel trees (with z coordinates).")
    for net, name, word in ((art, "arterial", "inlet"), (ven, "venous", "outlet")):
        if net.terminal_nodes.size == 0:
            raise InputError(f"The {name} tree has no free ends: every end is marked as "
                             f"an {word}, so it cannot exchange blood with the tissue.")
    return grid, art, ven


def solve(project, *, use_vessels, nonlinear=False, max_iter=100, tol=1e-10,
          phase_separation=False, elasticity=False, compliance=0.01):
    """Build the model (grid, trees, inlet/outlet nodes) and run the hybrid solve."""
    prm = project.params.to_core()
    if phase_separation:
        prm.fahraeus_lindqvist = True     # skimming only feeds back through F-L viscosity
    if elasticity:
        prm.compliance = compliance
    grid, art, ven = validate_run(project, use_vessels)
    domain_limit = project.grid.domain_limit
    if is_3d(project):                      # the 3-D hybrid: pressure, flux, time-of-flight
        from ..core.pipeline3d import run_hybrid_3d

        return run_hybrid_3d(prm, grid, art, ven, tuple(domain_limit))
    return run_hybrid(prm, grid, domain_limit=domain_limit, nonlinear=nonlinear,
                      art=art, ven=ven,
                      art_inlets=snap_io(art, project.inlets),
                      ven_outlets=snap_io(ven, project.outlets),
                      max_iter=max_iter, tol=tol,
                      phase_separation=phase_separation, elasticity=elasticity)


def solve_direct_model(project, use_vessels, rho_c=0.0):
    """Direct-connection model (Paper C): arterial & venous trees closed by
    resistive terminal links instead of a Darcy bed. Returns (art, ven, solution)."""
    from ..core.direct import connect_terminals, solve_direct

    prm = project.params.to_core()
    if prm.p_in == prm.p_out:
        raise InputError(f"p_in equals p_out ({prm.p_in}); there is no pressure drop to "
                         "drive flow. Set p_in above p_out in Parameters.")
    art, ven = build_networks(project, use_vessels, defaults=True)
    conns = connect_terminals(art, ven, rho_c=float(rho_c), viscosity=prm.mu)
    sol = solve_direct(art, ven, conns, prm.p_in, prm.p_out, viscosity=prm.mu)
    return art, ven, sol


def run_project(store, project, req: RunRequest) -> None:
    """Execute the simulation (hybrid Darcy or direct-connection) and persist it."""
    if getattr(req, "model", "hybrid") == "direct":
        art, ven, sol = solve_direct_model(project, req.use_vessels, req.rho_c)
        store.save_direct_result(project, art, ven, sol)
        return
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve(project, use_vessels=req.use_vessels, nonlinear=req.nonlinear,
                       max_iter=req.max_iter, tol=req.tol,
                       phase_separation=req.phase_separation,
                       elasticity=req.elasticity, compliance=req.compliance)
    notes = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    store.save_result(project, result, notes=notes or None)


def _check_sweep_params(project, names) -> None:
    """A 3-D project's sweeps can only vary what the 3-D hybrid depends on."""
    from ..uq import PARAMS_3D

    bad = [n for n in names if n not in PARAMS_3D]
    if is_3d(project) and bad:
        raise InputError(f"{', '.join(bad)} cannot vary on a 3-D project: its trees are "
                         "fixed and the 3-D solve uses one viscosity. Vary "
                         f"{', '.join(PARAMS_3D)} instead.")


def calibrate(project, target: float, param: str, bounds, qoi: str = "arterial_inflow") -> dict:
    """Recover ``param`` so the project model's ``qoi`` matches ``target`` (inverse
    problem). Returns the recovered value + achieved QoI as a JSON-ready dict."""
    from .. import inverse

    _check_sweep_params(project, [param])
    grid, art, ven = validate_run(project, use_vessels=True)
    base = project.params.model_dump()
    build = tree_builder(project)           # default trees rebuilt per sample if used
    res = inverse.calibrate(float(target), param, grid, tuple(bounds), qoi=qoi, base=base,
                            domain_limit=tuple(project.grid.domain_limit),
                            art=None if build else art, ven=None if build else ven,
                            trees=build)
    return {"param": res.param, "value": res.value, "target": res.target,
            "achieved": res.achieved, "qoi": res.qoi, "n_eval": res.n_eval}


def uncertainty(project, param_ranges: dict, n_samples: int = 32, seed: int = 0,
                method: str = "correlation") -> dict:
    """Uncertainty quantification over ``param_ranges`` on the project's grid/vasculature.

    ``method='correlation'`` returns QoI mean/std/percentiles + first-order Pearson
    sensitivities; ``method='sobol'`` returns variance-based Sobol first-order/total-effect
    indices (with the surrogate fit R²). JSON-ready dict either way."""
    from .. import uq

    _check_sweep_params(project, list(param_ranges))
    grid, art, ven = validate_run(project, use_vessels=True)
    base = project.params.model_dump()          # fixed Parameters fields
    ranges = {k: tuple(v) for k, v in param_ranges.items()}
    domain_limit = tuple(project.grid.domain_limit)
    build = tree_builder(project)           # default trees rebuilt per sample if used
    if build:
        art = ven = None
    if method == "sobol":
        res = uq.sobol_indices(ranges, grid, n_samples=n_samples, base=base, seed=seed,
                               domain_limit=domain_limit, art=art, ven=ven,
                               trees=build)
        return {"method": "sobol", "names": res.names, "qoi_names": res.qoi_names,
                "n_samples": res.n_samples, "first_order": res.first_order,
                "total_effect": res.total_effect, "fit_r2": res.fit_r2}
    res = uq.monte_carlo(ranges, grid, n_samples=n_samples, base=base, seed=seed,
                         domain_limit=domain_limit, art=art, ven=ven,
                         trees=build)
    return {"method": "correlation", "names": res.names, "qoi_names": res.qoi_names,
            "n_samples": int(n_samples), "mean": res.mean, "std": res.std,
            "p05": res.p05, "p95": res.p95, "sensitivity": res.sensitivity}


def adapt_network(project, target_exponent: float = 3.0, rate: float = 0.05,
                  n_iter: int = 60):
    """Structurally adapt the project's vessel radii toward Murray's law and store the
    remodelled arterial/venous trees. Returns ``(AdaptationResult, art, ven)``."""
    from ..core.pipeline import run_adaptation

    prm = project.params.to_core()
    grid, art, ven = validate_run(project, use_vessels=True)    # 2-D Grid or 3-D Grid3D
    ad = run_adaptation(prm, grid, target_exponent=target_exponent, rate=rate,
                        n_iter=n_iter, domain_limit=tuple(project.grid.domain_limit),
                        art=art, ven=ven)
    return ad


def multicompartment_field(project, field: str = "perfusion", n: int = 3,
                           beta: float = 1.0):
    """Solve an ``n``-compartment Darcy bed (arteriole→…→venule, uniform exchange
    ``beta``) and return one field: ``perfusion`` (the arteriole→next exchange, i.e.
    the capillary perfusion) or ``comp{c}`` (compartment ``c`` pressure). A 3-D
    project stacks 3-D compartments, so the field is ``(nx, ny, nz)``."""
    from ..core.pipeline import run_multicompartment

    n = int(np.clip(n, 2, 5))
    grid, art, ven = validate_run(project, use_vessels=True)
    prm = project.params.to_core()
    res = run_multicompartment(prm, grid, permeabilities=tuple([1.0] * n),
                               betas=tuple([float(beta)] * (n - 1)),
                               domain_limit=tuple(project.grid.domain_limit), art=art, ven=ven)
    if field == "perfusion":
        return res.perfusion[0], n
    if field.startswith("comp"):
        c = int(field[4:])
        if not 0 <= c < n:
            raise InputError(f"compartment {c} out of range for n={n}")
        return res.compartments[c], n
    raise InputError(f"field must be 'perfusion' or 'comp0'..'comp{n - 1}'")


def oxygen_field(project, field: str, k: float = 1.0, c_art: float = 1.0,
                 diffusivity: float = 0.0):
    """Re-solve the hybrid model and derive an oxygen/solute map (steady
    advection–reaction–diffusion). Returns ``(array, oef)`` for the requested field
    (``saturation`` / ``consumption`` / ``concentration``)."""
    from ..core import oxygen

    result = solve(project, use_vessels=True)
    ox = oxygen.oxygen_from_result(result, c_art=c_art, k=k, diffusivity=diffusivity)
    array = {"saturation": ox.saturation, "consumption": ox.extraction_rate,
             "concentration": ox.concentration}[field]
    return array, ox.oef


def run_network_tracer(store, project, req: TracerRequest) -> None:
    """Direct-model tracer: no grid, so animate the bolus along the vascular network
    using per-element transit times (arterial → connection → venous)."""
    from ..core import network_transit as nt

    art, ven, sol = solve_direct_model(project, use_vessels=True)
    anim = nt.transit_animation(art, ven, sol.connections, sol, n_frames=req.n_frames)
    store.save_network_tracer(project, **anim)


def run_tracer(store, project, req: TracerRequest) -> None:
    """Solve the flow, advect a contrast bolus, and persist the animation frames.

    The direct-connection model has no Darcy field, so it animates the bolus along
    the network (:func:`run_network_tracer`) instead of advecting it on the grid."""
    if store._direct_path(project.id).exists():
        run_network_tracer(store, project, req)
        return
    result = solve(project, use_vessels=True)

    # Time-scale from a characteristic transit time; long tail so the animation
    # shows inflow -> peak -> washout rather than cutting off.
    finite = result.tof[np.isfinite(result.tof)]
    t_char = float(np.percentile(finite, 80)) if finite.size else 5.0
    duration = req.duration or 5.0 * max(t_char, 1e-3)
    t0 = req.aif_t0 if req.aif_t0 is not None else 0.08 * duration

    n_steps = max(req.n_frames, 500)
    dt = duration / n_steps
    t = np.arange(n_steps) * dt
    aif = tkm.gamma_variate_aif(t, t0=t0, A=3.0, B=max(0.06 * duration, dt))
    peak = float(aif.max())
    if peak > 0:
        aif = aif / peak                        # unit-peak bolus -> concentrations O(1)
    full = transport.transport_forward(
        result.grid, result.fx, result.fy, result.sources,
        result.solution.terminal_flux, aif, dt, theta=1.0, fz=getattr(result, "fz", None))
    stride = max(1, n_steps // req.n_frames)
    frames, times = full[::stride][:req.n_frames], t[::stride][:req.n_frames]
    aif_frames = aif[::stride][:req.n_frames]        # AIF at the frame times, for TKM fitting
    # Robust colour scale: injection cells spike far above the tissue.
    active = frames[:, result.grid.mask]
    vmax = float(np.nanpercentile(active, 99)) if active.size else 0.0
    store.save_tracer(project, frames, times, vmax if vmax > 0 else 1.0, result.grid.mask,
                      aif=aif_frames)
