"""Time-of-flight and flux post-processing for the Darcy capillary bed.

Time-of-flight ``T`` (the time for a passive particle to travel from an arterial
terminal to a location) solves the steady advection equation ``v . grad T = phi``
(thesis Eqs. 17-21), discretised with a first-order upwind finite-volume scheme.
The upwind connectivity is shared with the tracer solver via
:func:`perfusion.darcy.upwind_operator`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .darcy import Grid, upwind_operator


def flux_magnitude(grid: Grid, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Cell-centred Darcy flux magnitude ``|v|`` ``(nx, ny)`` from face fluxes."""
    qx = 0.5 * (fx[0:grid.nx, :] + fx[1:grid.nx + 1, :])
    qy = 0.5 * (fy[:, 0:grid.ny] + fy[:, 1:grid.ny + 1])
    return np.hypot(qx, qy)


def _cell_sources(grid: Grid, sources, terminal_flux: np.ndarray) -> np.ndarray:
    """Signed per-cell source strength from mollified terminal fluxes."""
    q = np.zeros(grid.n_cells)
    for (cells, weights), flux in zip(sources, terminal_flux, strict=False):
        q[cells] += weights * flux
    return q


def time_of_flight(grid: Grid, fx: np.ndarray, fy: np.ndarray,
                   sources: list[tuple[np.ndarray, np.ndarray]],
                   terminal_flux: np.ndarray) -> np.ndarray:
    """Solve ``v . grad T = phi`` by first-order upwind finite volumes.

    Parameters
    ----------
    grid : Darcy grid.
    fx, fy : staggered face fluxes from :func:`perfusion.darcy.face_fluxes`.
    sources : per-terminal ``(cells, weights)`` from ``terminal_response``.
    terminal_flux : ``(T,)`` signed terminal fluxes aligned with ``sources``.

    Returns
    -------
    ``(nx, ny)`` time-of-flight field.
    """
    inflow_op, _ = upwind_operator(grid, fx, fy)          # (M @ c)[i] = inflow to i
    inflow = np.asarray(inflow_op.sum(axis=1)).ravel()
    injection = np.maximum(_cell_sources(grid, sources, terminal_flux), 0.0)

    diag = inflow + injection
    A = (-inflow_op + sp.diags(diag)).tocsr()

    dead = A.diagonal() == 0                               # cells with no inflow
    if dead.any():
        A = A + sp.diags(dead.astype(float))

    pore_volume = np.full(grid.n_cells, grid.cell_volume * grid.porosity)
    tof = spla.spsolve(A, pore_volume)
    tof[~grid.mask.ravel(order="F")] = np.nan   # inactive cells carry no meaning
    return tof.reshape(grid.nx, grid.ny, order="F")


def time_of_flight_3d(grid, fx, fy, fz, sources, terminal_flux) -> np.ndarray:
    """3-D time-of-flight ``v . grad T = phi`` (first-order upwind), the twin of
    :func:`time_of_flight` on a :class:`~perfusion.core.darcy3d.Grid3D` with the
    interior face fluxes of :func:`~perfusion.core.darcy3d.face_fluxes_3d`.
    Returns an ``(nx, ny, nz)`` field, NaN outside the mask."""
    from .darcy3d import upwind_operator_3d

    inflow_op, _ = upwind_operator_3d(grid, fx, fy, fz)
    inflow = np.asarray(inflow_op.sum(axis=1)).ravel()
    injection = np.maximum(_cell_sources(grid, sources, terminal_flux), 0.0)
    A = (-inflow_op + sp.diags(inflow + injection)).tocsr()
    dead = A.diagonal() == 0
    if dead.any():
        A = A + sp.diags(dead.astype(float))
    tof = spla.spsolve(A, np.full(grid.n_cells, grid.cell_volume * grid.porosity))
    tof[~grid.mask.ravel(order="F")] = np.nan
    return tof.reshape(grid.nx, grid.ny, grid.nz, order="F")
