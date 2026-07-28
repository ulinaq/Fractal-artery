"""End-to-end convenience pipeline tying the hybrid model together.

Given :class:`~perfusion.parameters.Parameters` and a :class:`~perfusion.darcy.Grid`,
build the arterial/venous trees, compute the Darcy terminal response, solve the
coupled flow, and return pressures, fluxes and time-of-flight in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import darcy, geometry, multicompartment, network, timeofflight
from .darcy import Grid
from .geometry import Network
from .parameters import Parameters
from .viscosity import apparent_viscosity


def _segment_viscosity(net: Network, prm: Parameters):
    """Per-segment viscosity: Fahraeus-Lindqvist if enabled, else the scalar mu."""
    if not prm.fahraeus_lindqvist:
        return prm.mu
    diameter_um = 2.0 * net.seg_radius * prm.radius_to_um
    return apparent_viscosity(diameter_um, prm.hematocrit, prm.mu)


@dataclass
class HybridResult:
    art: Network
    ven: Network
    grid: Grid
    solution: network.HybridSolution
    pressure: np.ndarray          # (nx, ny) capillary bed pressure field
    flux: np.ndarray              # (nx, ny) Darcy flux magnitude
    tof: np.ndarray               # (nx, ny) time-of-flight
    domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160)  # (xmin,xmax,ymin,ymax)
    # Intermediates reused by tracer transport (see perfusion.transport):
    fx: np.ndarray | None = None  # staggered x-face fluxes
    fy: np.ndarray | None = None  # staggered y-face fluxes
    sources: list = field(default_factory=list)  # per-terminal (cells, weights)


def _terminal_cells(grid: Grid, art: Network, ven: Network,
                    domain_limit: tuple[float, float, float, float],
                    strict: bool = False):
    """Map terminal node coordinates to Darcy grid cells (column-major linear).

    With ``strict=True`` a terminal falling outside the grid raises ``ValueError``
    (used by the optimiser to penalise ill-fitting trees); otherwise indices are
    clipped to the grid, which keeps general runs robust.
    """
    xmin, _, ymin, _ = domain_limit
    ta, tv = art.terminal_nodes, ven.terminal_nodes
    ix_a = np.ceil((art.node_x[ta] - xmin) / grid.hx).astype(int)
    iy_a = np.ceil((art.node_y[ta] - ymin) / grid.hy).astype(int)
    ix_v = np.floor((ven.node_x[tv] - xmin) / grid.hx).astype(int)
    iy_v = np.floor((ven.node_y[tv] - ymin) / grid.hy).astype(int)
    if strict:
        for ix, iy in ((ix_a, iy_a), (ix_v, iy_v)):
            if ix.min() < 0 or ix.max() >= grid.nx or iy.min() < 0 or iy.max() >= grid.ny:
                raise ValueError("terminal node falls outside the Darcy grid")
    ix_a, ix_v = np.clip(ix_a, 0, grid.nx - 1), np.clip(ix_v, 0, grid.nx - 1)
    iy_a, iy_v = np.clip(iy_a, 0, grid.ny - 1), np.clip(iy_v, 0, grid.ny - 1)
    cells = np.concatenate([grid.sub2ind(ix_a, iy_a), grid.sub2ind(ix_v, iy_v)])
    return _spread_coincident(grid, _snap_to_active(grid, cells))


def _snap_to_active(grid: Grid, cells: np.ndarray) -> np.ndarray:
    """Move terminals that land outside the mask to the nearest active cell.

    A terminal on an inactive cell gives a degenerate mollifier (empty support),
    which makes the Darcy coupling singular. Snapping to the nearest active cell
    keeps every terminal coupled to the tissue (organ-shaped or hand-drawn masks).
    """
    active = grid.mask.ravel(order="F")
    if active.all():
        return cells
    active_idx = np.flatnonzero(active)
    ax, ay = active_idx % grid.nx, active_idx // grid.nx
    cells = cells.copy()
    for i, c in enumerate(cells):
        if not active[c]:
            cx, cy = c % grid.nx, c // grid.nx
            cells[i] = int(active_idx[np.argmin((ax - cx) ** 2 + (ay - cy) ** 2)])
    return cells


def _spread_coincident(grid: Grid, cells: np.ndarray) -> np.ndarray:
    """Nudge coincident terminal cells to nearby free active cells.

    Two terminals mapping to the same cell give identical Darcy response columns
    (a singular coupling), so duplicates are moved to the nearest unused active
    neighbour (spiral search).
    """
    cells = cells.copy()
    taken = set()
    active = grid.mask.ravel(order="F")
    for i, c in enumerate(cells):
        if c not in taken:
            taken.add(int(c))
            continue
        ix, iy = int(c % grid.nx), int(c // grid.nx)
        for radius in range(1, max(grid.nx, grid.ny)):
            found = None
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < grid.nx and 0 <= jy < grid.ny:
                        j = int(grid.sub2ind(jx, jy))
                        if active[j] and j not in taken:
                            found = j
                            break
                if found is not None:
                    break
            if found is not None:
                cells[i] = found
                taken.add(found)
                break
    return cells


def run_hybrid(prm: Parameters, grid: Grid,
               domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
               eps: float = 4.0, nonlinear: bool = False,
               strict_domain: bool = False,
               art: Network | None = None, ven: Network | None = None,
               art_inlets=None, ven_outlets=None,
               max_iter: int = 100, tol: float = 1e-10,
               phase_separation: bool = False, elasticity: bool = False) -> HybridResult:
    """Build trees (or use the given ones), couple to Darcy, solve, post-process.

    Pass ``art``/``ven`` to use pre-built networks -- e.g. segmented from images
    via :func:`perfusion.segmentation.load_frog_system` -- instead of growing
    fractal trees from ``prm``. ``prm`` still supplies the flow boundary
    conditions and viscosity. ``strict_domain=True`` raises ``ValueError`` if a
    terminal falls outside the grid (see :func:`_terminal_cells`).

    ``art_inlets`` / ``ven_outlets`` are node-index arrays overriding which nodes
    carry the prescribed ``p_in`` / ``p_out`` (default: each tree's root). List
    several to model **multiple inlets/outlets**; the remaining leaves stay Darcy
    terminals.
    """
    if art is None:
        art = geometry.arterial_tree(prm)
    if ven is None:
        ven = geometry.venous_tree(prm)
    if art_inlets is not None:
        art = replace(art, inlet_nodes=np.asarray(art_inlets, dtype=int))
    if ven_outlets is not None:
        ven = replace(ven, inlet_nodes=np.asarray(ven_outlets, dtype=int))

    terminal_cells = _terminal_cells(grid, art, ven, domain_limit, strict=strict_domain)
    response, field_response, sources = darcy.terminal_response(grid, terminal_cells, eps)

    # Per-segment viscosity (radius-dependent when Fahraeus-Lindqvist is enabled).
    def _solve(a, v, visc):
        if nonlinear:
            return network.solve_nonlinear(a, v, response, prm.p_in, prm.p_out, prm, visc,
                                           max_iter=max_iter, tol=tol)
        return network.solve_linear(a, v, response, prm.p_in, prm.p_out, visc)

    if elasticity and prm.compliance:
        # Vessel-wall elasticity (Eq. 4.9): radii swell with transmural pressure,
        # r = r0 (1 + C (p - p_out)); resistance ∝ 1/r^4, so iterate to a fixed point.
        r0a, r0v = np.asarray(art.seg_radius, float), np.asarray(ven.seg_radius, float)
        art_e, ven_e = art, ven
        for _ in range(max(1, max_iter)):
            visc_e = (_segment_viscosity(art_e, prm), _segment_viscosity(ven_e, prm))
            sol = _solve(art_e, ven_e, visc_e)
            spa = 0.5 * (sol.p_a[art.seg_start] + sol.p_a[art.seg_end])
            spv = 0.5 * (sol.p_v[ven.seg_start] + sol.p_v[ven.seg_end])
            ra = np.clip(r0a * (1.0 + prm.compliance * (spa - prm.p_out)), 0.3 * r0a, 3.0 * r0a)
            rv = np.clip(r0v * (1.0 + prm.compliance * (spv - prm.p_out)), 0.3 * r0v, 3.0 * r0v)
            done = (np.max(np.abs(ra - art_e.seg_radius)) <= 1e-4 * float(r0a.max())
                    and np.max(np.abs(rv - ven_e.seg_radius)) <= 1e-4 * float(r0v.max()))
            art_e, ven_e = replace(art_e, seg_radius=ra), replace(ven_e, seg_radius=rv)
            if done:
                break
        art, ven = art_e, ven_e          # report the deformed geometry
    elif phase_separation and prm.fahraeus_lindqvist:
        # Couple RBC phase separation to viscosity: iterate arterial hematocrit
        # (Pries skimming) <-> Fahraeus-Lindqvist viscosity until it converges.
        from .hematocrit import segment_hematocrit

        d_art = 2.0 * art.seg_radius * prm.radius_to_um
        mu_ven = _segment_viscosity(ven, prm)
        hd = np.full(art.n_segments, prm.hematocrit)
        for _ in range(max(1, max_iter)):
            sol = _solve(art, ven, (apparent_viscosity(d_art, hd, prm.mu), mu_ven))
            hd_new = segment_hematocrit(art, np.abs(sol.q_a), prm.hematocrit, prm.radius_to_um)
            if np.max(np.abs(hd_new - hd)) <= 1e-4:
                hd = hd_new
                break
            hd = hd_new
    else:
        sol = _solve(art, ven, (_segment_viscosity(art, prm), _segment_viscosity(ven, prm)))

    pressure = darcy.reconstruct_field(field_response, sol.terminal_flux, balance=True)
    pressure = pressure.reshape(grid.nx, grid.ny, order="F")
    _, tx, ty = darcy.tpfa_operator(grid)
    fx, fy = darcy.face_fluxes(grid, pressure, tx, ty)
    tof = timeofflight.time_of_flight(grid, fx, fy, sources, sol.terminal_flux)
    flux = timeofflight.flux_magnitude(grid, fx, fy)

    return HybridResult(art=art, ven=ven, grid=grid, solution=sol,
                        pressure=pressure, flux=flux, tof=tof,
                        domain_limit=domain_limit, fx=fx, fy=fy, sources=sources)


@dataclass
class MultiCompartmentResult:
    art: Network
    ven: Network
    grid: Grid
    solution: network.HybridSolution
    permeabilities: tuple
    betas: tuple
    compartments: np.ndarray      # (N, nx, ny) per-compartment pressure fields
    perfusion: np.ndarray         # (N-1, nx, ny) inter-compartment exchange (capillary perfusion)
    domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160)


def run_multicompartment(prm: Parameters, grid: Grid,
                         permeabilities=(1.0, 1.0), betas=(1.0,),
                         domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
                         eps: float = 4.0, art: Network | None = None,
                         ven: Network | None = None, art_inlets=None,
                         ven_outlets=None) -> MultiCompartmentResult:
    """Couple the network to an ``N``-compartment Darcy bed (arteriole→…→venule).

    ``permeabilities`` is the per-compartment ``K_c`` (length ``N ≥ 1``) and ``betas``
    the exchange coefficients between adjacent compartments (length ``N−1``). Arterial
    terminals feed compartment 0; venous terminals drain compartment ``N−1``; the flow
    percolates down the stack through the exchange, whose per-cell field is the
    capillary perfusion. ``N = 1`` reduces exactly to :func:`run_hybrid`.
    """
    if len(betas) != len(permeabilities) - 1:
        raise ValueError("betas must have length len(permeabilities) - 1")
    if art is None:
        art = geometry.arterial_tree(prm)
    if ven is None:
        ven = geometry.venous_tree(prm)
    if art_inlets is not None:
        art = replace(art, inlet_nodes=np.asarray(art_inlets, dtype=int))
    if ven_outlets is not None:
        ven = replace(ven, inlet_nodes=np.asarray(ven_outlets, dtype=int))

    terminal_cells = _terminal_cells(grid, art, ven, domain_limit)
    n_a = len(art.terminal_nodes)
    mcr = multicompartment.terminal_response_multi(
        grid, terminal_cells[:n_a], terminal_cells[n_a:], permeabilities, betas, eps)
    visc = (_segment_viscosity(art, prm), _segment_viscosity(ven, prm))
    sol = network.solve_linear(art, ven, mcr.response, prm.p_in, prm.p_out, visc)

    comp_flat = multicompartment.compartment_fields(mcr, sol.terminal_flux, balance=True)
    compartments = np.stack([comp_flat[c].reshape(grid.nx, grid.ny, order="F")
                             for c in range(mcr.n_compartments)])
    perfusion = multicompartment.exchange_field(grid, comp_flat, betas)
    return MultiCompartmentResult(
        art=art, ven=ven, grid=grid, solution=sol,
        permeabilities=tuple(permeabilities), betas=tuple(betas),
        compartments=compartments, perfusion=perfusion, domain_limit=domain_limit)
