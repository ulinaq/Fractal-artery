"""End-to-end convenience pipeline tying the hybrid model together.

Given :class:`~perfusion.parameters.Parameters` and a :class:`~perfusion.darcy.Grid`,
build the arterial/venous trees, compute the Darcy terminal response, solve the
coupled flow, and return pressures, fluxes and time-of-flight in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import adaptation, darcy, geometry, multicompartment, network, timeofflight
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
    off = np.flatnonzero(~active[cells])
    if not off.size:
        return cells
    from scipy.spatial import cKDTree

    active_idx = np.flatnonzero(active)
    pts = np.column_stack([active_idx % grid.nx, active_idx // grid.nx])
    q = np.column_stack([cells[off] % grid.nx, cells[off] // grid.nx])
    k = min(8, len(active_idx))
    dist, nbr = cKDTree(pts).query(q, k=k)
    dist, nbr = dist.reshape(len(off), k), nbr.reshape(len(off), k)
    # ties (equal integer distances) -> lowest active index, as a brute argmin would pick
    tie = np.where(np.isclose(dist, dist[:, :1]), nbr, np.iinfo(np.int64).max)
    cells = cells.copy()
    cells[off] = active_idx[tie.min(axis=1)]
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


# above this many dense entries (n_cells × terminals ≈ 8 bytes each) "auto" switches
# from the Schur response to the sparse monolithic solve (~200 MB of dense block)
MONOLITHIC_AUTO_ENTRIES = 25_000_000


def use_monolithic(coupling: str, n_cells: int, n_terminals: int) -> bool:
    """Whether a solve should use the monolithic coupling (see :func:`run_hybrid`)."""
    if coupling not in ("auto", "schur", "monolithic"):
        raise ValueError("coupling must be 'auto', 'schur' or 'monolithic'")
    return coupling == "monolithic" or (
        coupling == "auto" and n_cells * n_terminals > MONOLITHIC_AUTO_ENTRIES)


def darcy_response_for(prm: Parameters, grid: Grid,
                       domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
                       art: Network | None = None, ven: Network | None = None,
                       eps: float = 4.0):
    """``(art, ven, response)`` for reusing the Darcy factorisation across solves.

    Builds the trees (from ``prm`` unless given) and the terminal response once; pass
    ``art``/``ven``/``darcy_response=response`` to :func:`run_hybrid` in a loop that
    keeps the grid (incl. permeability / mask), the terminal positions and ``eps``
    fixed — e.g. sweeps over ``p_in``, ``p_out``, ``mu`` or hematocrit. Results are
    identical to recomputing; only the dominant LU + back-substitution is skipped.
    """
    if art is None:
        art = geometry.arterial_tree(prm)
    if ven is None:
        ven = geometry.venous_tree(prm)
    cells = _terminal_cells(grid, art, ven, domain_limit)
    return art, ven, darcy.terminal_response(grid, cells, eps)


def run_hybrid(prm: Parameters, grid: Grid,
               domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
               eps: float = 4.0, nonlinear: bool = False,
               strict_domain: bool = False,
               art: Network | None = None, ven: Network | None = None,
               art_inlets=None, ven_outlets=None,
               max_iter: int = 100, tol: float = 1e-10,
               phase_separation: bool = False, elasticity: bool = False,
               darcy_response=None, coupling: str = "auto") -> HybridResult:
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

    ``darcy_response`` optionally supplies a precomputed ``(response, field_response,
    sources)`` from :func:`darcy.terminal_response`, skipping the dominant Darcy
    factorisation. Reuse it only when the grid, terminal *positions* and ``eps`` are
    unchanged (e.g. an outer loop that varies only ``p_in``/``mu`` or the segment
    *radii*, as in :func:`run_adaptation`) — the Darcy response is independent of both.

    ``coupling`` chooses how the Darcy bed enters the network solve: ``"schur"`` —
    the dense terminal response (``O(n_cells · T)`` memory, fastest for small
    problems, and reusable across solves); ``"monolithic"`` — one sparse system with
    the bed pressures as unknowns (``O(nnz)`` memory: fine grids, many terminals);
    ``"auto"`` (default) picks monolithic only when the dense block would exceed
    ``MONOLITHIC_AUTO_ENTRIES``. Both give the same solution.
    """
    if art is None:
        art = geometry.arterial_tree(prm)
    if ven is None:
        ven = geometry.venous_tree(prm)
    if art_inlets is not None:
        art = replace(art, inlet_nodes=np.asarray(art_inlets, dtype=int))
    if ven_outlets is not None:
        ven = replace(ven, inlet_nodes=np.asarray(ven_outlets, dtype=int))

    if darcy_response is not None:
        response, field_response, sources = darcy_response
    else:
        terminal_cells = _terminal_cells(grid, art, ven, domain_limit, strict=strict_domain)
        if use_monolithic(coupling, grid.n_cells, len(terminal_cells)):
            response, sources = darcy.coupling_operator(grid, terminal_cells, eps)
            field_response = None              # the bed pressure comes from the solve
        else:
            response, field_response, sources = darcy.terminal_response(grid, terminal_cells,
                                                                        eps)

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

    if field_response is None:                 # monolithic: the bed is part of the solve
        pressure = np.asarray(sol.darcy_pressure)
    else:
        pressure = darcy.reconstruct_field(field_response, sol.terminal_flux, balance=True)
    pressure = pressure.reshape(grid.nx, grid.ny, order="F")
    tx, ty = darcy.transmissibilities(grid)     # cheaper than rebuilding tpfa_operator for tx/ty
    fx, fy = darcy.face_fluxes(grid, pressure, tx, ty)
    tof = timeofflight.time_of_flight(grid, fx, fy, sources, sol.terminal_flux)
    flux = timeofflight.flux_magnitude(grid, fx, fy)

    return HybridResult(art=art, ven=ven, grid=grid, solution=sol,
                        pressure=pressure, flux=flux, tof=tof,
                        domain_limit=domain_limit, fx=fx, fy=fy, sources=sources)


@dataclass
class AdaptationResult:
    art: Network                  # adapted arterial tree (radii evolved)
    ven: Network                  # adapted venous tree
    result: HybridResult          # final hybrid solve on the adapted network
    murray_history: list          # fitted |Q| ∝ r^k exponent per iteration
    murray_exponent: float        # exponent of the returned (best) network (→ target)
    murray_start: float           # exponent before adaptation
    n_iter: int


def run_adaptation(prm: Parameters, grid, target_exponent: float = 3.0,
                   domain_limit: tuple = (-70, 70, 0, 160),
                   rate: float = 0.05, n_iter: int = 60,
                   art: Network | None = None, ven: Network | None = None,
                   eps: float | None = None) -> AdaptationResult:
    """Structurally adapt vessel radii toward Murray's law (reduced Pries–Secomb).

    Iterates: solve the hybrid flow → compute each segment's wall shear → nudge radii
    toward a fixed set-point (:func:`adaptation.adapt_radii`, target = the initial mean
    wall shear). Shear-driven remodelling monotonically drives the ``|Q| ∝ r^k`` scaling
    exponent from its fractal starting value toward ``k = 3`` (Murray's law), the classic
    emergent optimum; since the pure-shear stimulus keeps over-differentiating past it,
    the loop **early-stops at the closest approach** to ``target_exponent`` and returns
    that Murray-optimal network. Radii are bounded to ``[0.1, 10]×`` their initial value.

    ``grid`` may be a 2-D :class:`Grid` or a :class:`~perfusion.core.darcy3d.Grid3D`
    (then ``art``/``ven`` are 3-D trees and ``domain_limit`` the 6-value window);
    ``eps`` defaults to each path's mollifier radius (4 in 2-D, 2 in 3-D).
    """
    art = geometry.arterial_tree(prm) if art is None else art
    ven = geometry.venous_tree(prm) if ven is None else ven
    r0a, r0v = np.asarray(art.seg_radius, float), np.asarray(ven.seg_radius, float)

    def exponent(a, v, sol):
        return adaptation.murray_exponent(
            np.concatenate([a.seg_radius, v.seg_radius]),
            np.concatenate([sol.q_a, sol.q_v]))

    # Adaptation only changes radii, never terminal positions, so the Darcy terminal
    # response is constant across every iteration — compute it once and reuse it (the
    # dominant per-solve cost); only the network resistances change and are re-solved.
    if isinstance(grid, Grid):
        eps = 4.0 if eps is None else eps
        resp = darcy.terminal_response(grid, _terminal_cells(grid, art, ven, domain_limit), eps)

        def solve(a, v):
            return run_hybrid(prm, grid, domain_limit=domain_limit, eps=eps, art=a, ven=v,
                              darcy_response=resp)
    else:                                            # 3-D: same loop, 3-D Darcy coupling
        from .pipeline3d import darcy_coupling_3d, run_hybrid_3d

        eps = 2.0 if eps is None else eps
        coupling = darcy_coupling_3d(grid, art, ven, domain_limit, eps)

        def solve(a, v):
            return run_hybrid_3d(prm, grid, a, v, domain_limit, eps=eps, darcy=coupling)

    # fixed shear set-point = the initial mean wall shear (on-scale, so radii don't drift uniformly)
    first = solve(art, ven)
    tau0 = adaptation.wall_shear_stress(art.seg_radius, first.solution.q_a, prm.mu)
    tau_ref = float(tau0[tau0 > 0].mean()) if np.any(tau0 > 0) else 1.0

    history: list[float] = []
    best = None
    best_err = np.inf
    result = first
    for it in range(max(1, n_iter)):
        result = solve(art, ven)
        k = exponent(art, ven, result.solution)
        history.append(k)
        err = abs(k - target_exponent)
        if err < best_err:
            best_err, best = err, (art, ven, result, k, it)
        elif it - best[4] >= 3:                     # 3 iters past the closest approach → stop
            break
        ra, _ = adaptation.adapt_radii(art.seg_radius, result.solution.q_a, tau_ref, prm.mu,
                                       rate=rate, r_min=0.1 * r0a, r_max=10 * r0a)
        rv, _ = adaptation.adapt_radii(ven.seg_radius, result.solution.q_v, tau_ref, prm.mu,
                                       rate=rate, r_min=0.1 * r0v, r_max=10 * r0v)
        art, ven = replace(art, seg_radius=ra), replace(ven, seg_radius=rv)

    b_art, b_ven, b_result, b_k, _ = best
    return AdaptationResult(art=b_art, ven=b_ven, result=b_result, murray_history=history,
                            murray_exponent=b_k, murray_start=history[0], n_iter=len(history))


@dataclass
class MultiCompartmentResult:
    art: Network
    ven: Network
    grid: Grid
    solution: network.HybridSolution
    permeabilities: tuple
    betas: tuple
    compartments: np.ndarray      # (N, nx, ny[, nz]) per-compartment pressure fields
    perfusion: np.ndarray         # (N-1, nx, ny[, nz]) inter-compartment exchange (perfusion)
    domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160)


def run_multicompartment(prm: Parameters, grid: Grid,
                         permeabilities=(1.0, 1.0), betas=(1.0,),
                         domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
                         eps: float | None = None, art: Network | None = None,
                         ven: Network | None = None, art_inlets=None,
                         ven_outlets=None) -> MultiCompartmentResult:
    """Couple the network to an ``N``-compartment Darcy bed (arteriole→…→venule).

    ``permeabilities`` is the per-compartment ``K_c`` (length ``N ≥ 1``) and ``betas``
    the exchange coefficients between adjacent compartments (length ``N−1``). Arterial
    terminals feed compartment 0; venous terminals drain compartment ``N−1``; the flow
    percolates down the stack through the exchange, whose per-cell field is the
    capillary perfusion. ``N = 1`` reduces exactly to :func:`run_hybrid`.

    A 3-D :class:`~perfusion.core.darcy3d.Grid3D` (with 3-D ``art``/``ven`` and a
    6-value ``domain_limit``) stacks 3-D compartments the same way, using the scalar
    viscosity like :func:`~perfusion.core.pipeline3d.run_hybrid_3d`; fields are then
    ``(N, nx, ny, nz)``. ``eps`` defaults to 4 cells in 2-D and 2 in 3-D (as the
    single-bed solvers).
    """
    three_d = getattr(grid, "nz", None) is not None
    if eps is None:
        eps = 2.0 if three_d else 4.0
    if three_d and (art is None or ven is None):
        raise ValueError("a 3-D multi-compartment solve needs the 3-D trees (art, ven)")
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

    if three_d:
        from .pipeline3d import _terminal_cells_3d

        terminal_cells = _terminal_cells_3d(grid, art, ven, domain_limit)
        visc = prm.mu
    else:
        terminal_cells = _terminal_cells(grid, art, ven, domain_limit)
        visc = (_segment_viscosity(art, prm), _segment_viscosity(ven, prm))
    n_a = len(art.terminal_nodes)
    mcr = multicompartment.terminal_response_multi(
        grid, terminal_cells[:n_a], terminal_cells[n_a:], permeabilities, betas, eps)
    sol = network.solve_linear(art, ven, mcr.response, prm.p_in, prm.p_out, visc)

    comp_flat = multicompartment.compartment_fields(mcr, sol.terminal_flux, balance=True)
    compartments = np.stack([comp_flat[c].reshape(grid.mask.shape, order="F")
                             for c in range(mcr.n_compartments)])
    perfusion = multicompartment.exchange_field(grid, comp_flat, betas)
    return MultiCompartmentResult(
        art=art, ven=ven, grid=grid, solution=sol,
        permeabilities=tuple(permeabilities), betas=tuple(betas),
        compartments=compartments, perfusion=perfusion, domain_limit=domain_limit)
