"""Model building + job execution, independent of the web app.

Kept out of ``app.py`` so both the FastAPI endpoints and background workers (the
thread pool *and* an optional RQ worker process) can build the model and run a
project without importing the app. Functions take a ``ProjectStore`` + a
``Project`` and persist their results through the store.
"""

from __future__ import annotations

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


def build_grid(project):
    """Grid + its domain window, with the tissue mask: a stored raster mask if
    present (shape must match the grid), else the drawn tissue polygons, else none."""
    nx, ny, domain_limit = project.grid.nx, project.grid.ny, project.grid.domain_limit
    raster = getattr(project, "mask", None)
    if raster is not None and np.asarray(raster).shape == (nx, ny):
        mask = np.asarray(raster, dtype=bool)
    elif project.regions:
        mask = rasterize_mask(nx, ny, domain_limit, project.regions)
    else:
        mask = None
    return project.grid.to_core(mask=mask), domain_limit


def build_networks(project, use_vessels):
    """Arterial/venous networks: a grown/segmented tree wins, else hand-drawn."""
    art = ven = None
    if use_vessels and project.has_vessels:
        art = polylines_to_network(project.arteries) if project.arteries else None
        ven = polylines_to_network(project.veins) if project.veins else None
    if project.artery_network is not None:
        art = project.artery_network
    if project.vein_network is not None:
        ven = project.vein_network
    return art, ven


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

    art, ven = build_networks(project, use_vessels)
    if project.inlets and art is None:
        art = arterial_tree(prm)
    if project.outlets and ven is None:
        ven = venous_tree(prm)
    # art/ven None is fine: run_hybrid falls back to the default fractal trees.
    if prm.p_in == prm.p_out:
        raise InputError(
            f"p_in equals p_out ({prm.p_in}); there is no pressure drop to drive flow. "
            "Set p_in above p_out in Parameters.")
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
    art, ven = build_networks(project, use_vessels)
    art = art if art is not None else arterial_tree(prm)
    ven = ven if ven is not None else venous_tree(prm)
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
        result.solution.terminal_flux, aif, dt, theta=1.0)
    stride = max(1, n_steps // req.n_frames)
    frames, times = full[::stride][:req.n_frames], t[::stride][:req.n_frames]
    aif_frames = aif[::stride][:req.n_frames]        # AIF at the frame times, for TKM fitting
    # Robust colour scale: injection cells spike far above the tissue.
    active = frames[:, result.grid.mask]
    vmax = float(np.nanpercentile(active, 99)) if active.size else 0.0
    store.save_tracer(project, frames, times, vmax if vmax > 0 else 1.0, result.grid.mask,
                      aif=aif_frames)
