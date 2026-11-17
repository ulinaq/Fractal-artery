"""Passive tracer transport in the capillary bed.

Advective transport of a contrast agent along the bloodstream (diffusion
neglected), integrated in time with an explicit theta rule (thesis Eq. 5.7).
Used to synthesise DCE-MRI-like tissue concentration curves (Papers B and C).
Shares the upwind connectivity with the time-of-flight solver via
:func:`perfusion.darcy.upwind_operator`.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .darcy import Grid, upwind_operator


def _upwind(grid, fx, fy, fz=None):
    """The upwind operator for a 2-D grid, or a 3-D grid when ``fz`` is given."""
    if fz is None:
        return upwind_operator(grid, fx, fy)
    from .darcy3d import upwind_operator_3d

    return upwind_operator_3d(grid, fx, fy, fz)


def build_advection(grid: Grid, fx: np.ndarray, fy: np.ndarray, sources,
                    terminal_flux: np.ndarray, fz: np.ndarray | None = None):
    """Assemble the linear advection operator and the source/sink split.

    Returns ``(A, inject, extract, pore_volume)`` such that the semi-discrete
    transport is

        pore_volume * dC/dt = A @ C + inject * ca(t) - extract * C,

    where ``A`` gathers net inter-cell upwind fluxes (inflow minus outflow),
    ``inject`` is the arterial injection weight per cell, and ``extract`` the
    venous extraction weight. Passing ``fz`` (and a ``Grid3D``) assembles the
    3-D operator over the interior faces of ``darcy3d.face_fluxes_3d``.
    """
    inflow_op, outflow = _upwind(grid, fx, fy, fz)
    A = inflow_op - sp.diags(outflow)

    inject = np.zeros(grid.n_cells)
    extract = np.zeros(grid.n_cells)
    for (cells, weights), flux in zip(sources, terminal_flux, strict=False):
        if flux >= 0:
            inject[cells] += weights * flux
        else:
            extract[cells] += weights * (-flux)

    pore_volume = np.full(grid.n_cells, grid.cell_volume * grid.porosity)
    return A.tocsr(), inject, extract, pore_volume


def cfl_limit(grid: Grid, fx, fy, sources, terminal_flux, fz=None) -> float:
    """Largest stable explicit-Euler time step (min cell volume / total outflux)."""
    A, _, extract, pore_volume = build_advection(grid, fx, fy, sources, terminal_flux, fz)
    outflux = -A.diagonal() + extract           # diag(A) = -face outflow
    active = outflux > 0
    return float(np.min(pore_volume[active] / outflux[active])) if active.any() else np.inf


def transport_forward(grid: Grid, fx, fy, sources, terminal_flux,
                      aif: np.ndarray, dt: float, theta: float = 0.5,
                      clip_negative: bool = True, fz=None) -> np.ndarray:
    """Tracer transport via the theta rule, returning concentration snapshots.

    Parameters
    ----------
    aif : arterial input function sampled at ``dt`` (length = number of steps).
    dt : time step.
    theta : time-integration weight (thesis Eq. 5.7). ``0`` = explicit Euler
        (conditionally stable -- a CFL warning is issued if ``dt`` is too large),
        ``0.5`` = Crank-Nicolson (default, 2nd order), ``1`` = implicit Euler
        (unconditionally stable). For ``theta > 0`` a single LU factorisation of
        ``I - theta*dt*L`` is reused across all steps.
    clip_negative : clamp tiny negative concentrations to zero.
    fz : z-face fluxes — pass them (with a ``Grid3D``) for 3-D transport.

    Returns
    -------
    ``(n_steps, nx, ny)`` (or ``(n_steps, nx, ny, nz)``) tissue concentration fields.
    """
    A, inject, extract, pore_volume = build_advection(grid, fx, fy, sources,
                                                      terminal_flux, fz)
    n = grid.n_cells
    inv_volume = 1.0 / pore_volume
    # semi-discrete: dC/dt = L @ C + inv_volume * inject * ca(t)
    L = sp.diags(inv_volume) @ (A - sp.diags(extract))
    source = inv_volume * inject

    if theta == 0.0:
        limit = cfl_limit(grid, fx, fy, sources, terminal_flux, fz)
        if dt > limit:
            warnings.warn(
                f"transport_forward: explicit dt={dt:g} exceeds the CFL limit "
                f"{limit:g}; use theta>0 or a smaller dt", RuntimeWarning, stacklevel=2)
        lhs_solve = None
        rhs_op = sp.identity(n, format="csr") + dt * L
    else:
        identity = sp.identity(n, format="csr")
        lhs_solve = spla.splu((identity - theta * dt * L).tocsc()).solve
        rhs_op = identity + (1.0 - theta) * dt * L

    concentration = np.zeros(n)
    shape = grid.mask.shape
    frames = np.empty((len(aif), *shape))
    for step, ca in enumerate(aif):
        ca_next = aif[step + 1] if step + 1 < len(aif) else ca
        ca_eff = theta * ca_next + (1.0 - theta) * ca
        rhs = rhs_op @ concentration + dt * source * ca_eff
        concentration = rhs if lhs_solve is None else lhs_solve(rhs)
        if clip_negative:
            concentration = np.clip(concentration, 0.0, None)
        frames[step] = concentration.reshape(shape, order="F")
    return frames
