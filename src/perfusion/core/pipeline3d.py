"""3-D hybrid pipeline: couple 3-D vascular trees to a 3-D Darcy bed.

The 3-D analogue of :func:`perfusion.core.pipeline.run_hybrid`. Because the
network solver (:func:`perfusion.core.network.solve_linear`) is coordinate-free
and :func:`perfusion.core.darcy.reconstruct_field` is dimension-agnostic, this
only adds the 3-D-specific pieces: mapping terminal nodes to 3-D grid cells and
the 3-D Darcy terminal response (:func:`perfusion.core.darcy3d.terminal_response_3d`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import darcy3d, network
from .darcy import reconstruct_field
from .darcy3d import Grid3D
from .geometry import Network
from .parameters import Parameters
from .timeofflight import time_of_flight_3d


@dataclass
class HybridResult3D:
    art: Network
    ven: Network
    grid: Grid3D
    solution: network.HybridSolution
    pressure: np.ndarray                       # (nx, ny, nz) capillary-bed pressure
    domain_limit: tuple                        # (xmin, xmax, ymin, ymax, zmin, zmax)
    flux: np.ndarray | None = None             # (nx, ny, nz) Darcy flux magnitude
    tof: np.ndarray | None = None              # (nx, ny, nz) time-of-flight


def _spread_coincident_3d(grid: Grid3D, cells: np.ndarray) -> np.ndarray:
    """Nudge coincident terminal cells to nearby free active cells (3-D search)."""
    cells = cells.copy()
    active = grid.mask.ravel(order="F")
    taken: set[int] = set()
    nx, ny = grid.nx, grid.ny
    for i, c in enumerate(cells):
        if c not in taken:
            taken.add(int(c))
            continue
        ix, iy, iz = int(c % nx), int((c // nx) % ny), int(c // (nx * ny))
        found = None
        for radius in range(1, max(nx, ny, grid.nz)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        jx, jy, jz = ix + dx, iy + dy, iz + dz
                        if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < grid.nz:
                            j = int(grid.sub2ind(jx, jy, jz))
                            if active[j] and j not in taken:
                                found = j
                                break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            cells[i] = found
            taken.add(found)
    return cells


def _terminal_cells_3d(grid, art, ven, domain_limit):
    """Map arterial+venous terminal nodes to column-major 3-D grid cells."""
    xmin, _, ymin, _, zmin, _ = domain_limit

    def cells_of(net):
        t = net.terminal_nodes
        ix = np.clip(np.floor((net.node_x[t] - xmin) / grid.hx).astype(int), 0, grid.nx - 1)
        iy = np.clip(np.floor((net.node_y[t] - ymin) / grid.hy).astype(int), 0, grid.ny - 1)
        iz = np.clip(np.floor((net.node_z[t] - zmin) / grid.hz).astype(int), 0, grid.nz - 1)
        return grid.sub2ind(ix, iy, iz)

    cells = np.concatenate([cells_of(art), cells_of(ven)])
    return _spread_coincident_3d(grid, _snap_to_active_3d(grid, cells))


def _snap_to_active_3d(grid: Grid3D, cells: np.ndarray) -> np.ndarray:
    """Move terminals that land outside the mask to the nearest active cell (the
    3-D twin of ``pipeline._snap_to_active``): a terminal on an inactive cell has
    an empty mollifier support and would make the coupling singular."""
    active = grid.mask.ravel(order="F")
    if active.all():
        return cells
    from scipy.spatial import cKDTree

    nx, ny = grid.nx, grid.ny
    def ijk(c):
        return np.column_stack([c % nx, (c // nx) % ny, c // (nx * ny)])
    active_idx = np.flatnonzero(active)
    cells = cells.copy()
    off = ~active[cells]
    if off.any():
        _, j = cKDTree(ijk(active_idx)).query(ijk(cells[off]))
        cells[off] = active_idx[j]
    return cells


def run_hybrid_3d(prm: Parameters, grid: Grid3D, art: Network, ven: Network,
                  domain_limit, eps: float = 2.0,
                  art_inlets=None, ven_outlets=None) -> HybridResult3D:
    """Couple 3-D arterial/venous trees to the 3-D Darcy bed and solve.

    ``art``/``ven`` are 3-D networks (``node_z`` set, e.g. from
    :func:`perfusion.core.geometry.grow_tree_3d`); ``domain_limit`` is the 3-D
    window ``(xmin, xmax, ymin, ymax, zmin, zmax)`` the grid maps to. ``prm``
    supplies ``p_in``/``p_out`` and the (scalar) viscosity ``mu``.
    """
    if art_inlets is not None:
        art = replace(art, inlet_nodes=np.asarray(art_inlets, dtype=int))
    if ven_outlets is not None:
        ven = replace(ven, inlet_nodes=np.asarray(ven_outlets, dtype=int))

    terminal_cells = _terminal_cells_3d(grid, art, ven, domain_limit)
    response, field_response, sources = darcy3d.terminal_response_3d(grid, terminal_cells, eps)
    sol = network.solve_linear(art, ven, response, prm.p_in, prm.p_out, prm.mu)
    pressure = reconstruct_field(field_response, sol.terminal_flux, balance=True)
    pressure = pressure.reshape(grid.nx, grid.ny, grid.nz, order="F")
    # the same post-processing as 2-D: Darcy flux magnitude + time-of-flight
    fx, fy, fz = darcy3d.face_fluxes_3d(grid, pressure)
    flux = darcy3d.flux_magnitude_3d(grid, fx, fy, fz)
    tof = time_of_flight_3d(grid, fx, fy, fz, sources, sol.terminal_flux)
    return HybridResult3D(art=art, ven=ven, grid=grid, solution=sol,
                          pressure=pressure, domain_limit=tuple(domain_limit),
                          flux=flux, tof=tof)
