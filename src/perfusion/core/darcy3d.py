"""3-D structured Darcy grid + TPFA operator (Phase 4 foundation).

Extends the two-point flux approximation to 3-D: a uniform ``(nx, ny, nz)`` grid
with column-major (x-fastest) linear indexing ``ix + iy*nx + iz*nx*ny``,
face transmissibilities ``T = k * face_area / spacing``, a mask that hydraulically
isolates inactive cells, and a singular Neumann operator (pin one cell). Verified
against an analytic linear pressure ramp (see ``tests/test_darcy3d.py``).

This is the compute foundation for 3-D trees / segmentation / vtk.js rendering;
it does **not** yet couple to the 1-D network solver — that is the rest of Phase
4. Kept in its own module so the validated 2-D engine (``darcy.py``) is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Grid3D:
    """A uniform 3-D Cartesian grid for the Darcy bed."""

    nx: int
    ny: int
    nz: int
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    permeability: float = 1.0
    thickness: float = 1.0                 # unused in 3-D; kept for API symmetry
    mask: np.ndarray | None = None         # (nx, ny, nz) bool; True = active

    hx: float = field(init=False)
    hy: float = field(init=False)
    hz: float = field(init=False)

    def __post_init__(self) -> None:
        self.hx = self.lx / self.nx
        self.hy = self.ly / self.ny
        self.hz = self.lz / self.nz
        if self.mask is None:
            self.mask = np.ones((self.nx, self.ny, self.nz), dtype=bool)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float:
        return self.hx * self.hy * self.hz

    def sub2ind(self, ix, iy, iz) -> np.ndarray:
        """Column-major (x-fastest) linear index, matching ``reshape(order='F')``."""
        return (np.asarray(ix) + np.asarray(iy) * self.nx
                + np.asarray(iz) * self.nx * self.ny)

    def cell_centers(self):
        xc = (np.arange(self.nx) + 0.5) * self.hx
        yc = (np.arange(self.ny) + 0.5) * self.hy
        zc = (np.arange(self.nz) + 0.5) * self.hz
        gx, gy, gz = np.meshgrid(xc, yc, zc, indexing="ij")
        return (gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F"))


def _face_transmissibilities(grid: Grid3D):
    """Per-direction face transmissibility (scalar; uniform grid + scalar K)."""
    k = grid.permeability
    tx = k * grid.hy * grid.hz / grid.hx
    ty = k * grid.hx * grid.hz / grid.hy
    tz = k * grid.hx * grid.hy / grid.hz
    return tx, ty, tz


def tpfa_operator_3d(grid: Grid3D) -> sp.csr_matrix:
    """Assemble the 3-D TPFA matrix for ``-div(K grad p) = q``.

    Faces touching an inactive cell carry no flow; inactive cells get identity
    rows so the active domain is isolated. The operator is singular (Neumann) —
    pin one cell before solving.
    """
    nx, ny, nz, n = grid.nx, grid.ny, grid.nz, grid.n_cells
    active = grid.mask.ravel(order="F")
    cell = np.arange(n).reshape(nx, ny, nz, order="F")
    tx, ty, tz = _face_transmissibilities(grid)

    faces = [
        (cell[:-1, :, :], cell[1:, :, :], tx),
        (cell[:, :-1, :], cell[:, 1:, :], ty),
        (cell[:, :, :-1], cell[:, :, 1:], tz),
    ]
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for owner, neigh, t in faces:
        o, m = owner.ravel(order="F"), neigh.ravel(order="F")
        keep = active[o] & active[m]                 # both cells active
        o, m = o[keep], m[keep]
        rows.extend((o, m))
        cols.extend((m, o))
        vals.extend((np.full(o.size, -t), np.full(m.size, -t)))
        np.add.at(diag, o, t)
        np.add.at(diag, m, t)

    idx = np.arange(n)
    diag = np.where(active, diag, 1.0)               # inactive -> identity diagonal
    rows.append(idx)
    cols.append(idx)
    vals.append(diag)
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    return A.tocsr()


def point_source_3d(grid: Grid3D, center: int, eps: float):
    """Compactly supported 3-D source around a cell (the thesis bump kernel).

    Mirrors the 2-D ``darcy.point_source``: ``exp(-1/(1-r^2))`` over cells within
    ``eps`` cell-radii of ``center``, restricted to the active mask, weights
    summing to 1. Returns ``(cell_indices, weights)``.
    """
    xc, yc, zc = grid.cell_centers()
    dist2 = (((xc - xc[center]) / grid.hx) ** 2
             + ((yc - yc[center]) / grid.hy) ** 2
             + ((zc - zc[center]) / grid.hz) ** 2)
    support = (dist2 <= eps**2) & grid.mask.ravel(order="F")
    if not support.any():
        support[center] = True
    normalised = np.clip(dist2[support] / eps**2, 0.0, 1.0 - 1e-12)
    weights = np.exp(-1.0 / (1.0 - normalised))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones_like(weights)
        total = weights.sum()
    weights /= total
    return np.flatnonzero(support), weights


def solve_pressure_3d(grid: Grid3D, source: np.ndarray) -> np.ndarray:
    """Solve ``-div(K grad p) = source`` on the active 3-D domain.

    ``source`` is the net volumetric injection per cell (must sum to ~0 over the
    active domain for a well-posed Neumann problem). One active cell is pinned as
    the pressure datum. Returns the ``(nx, ny, nz)`` pressure field.
    """
    active = grid.mask.ravel(order="F")
    A = tpfa_operator_3d(grid).tolil()
    b = np.asarray(source, dtype=float).copy()
    pin = int(np.flatnonzero(active)[0])
    A.rows[pin], A.data[pin], b[pin] = [pin], [1.0], 0.0     # datum: p_pin = 0
    p = spla.spsolve(A.tocsr(), b)
    return p.reshape(grid.nx, grid.ny, grid.nz, order="F")


def face_fluxes_3d(grid: Grid3D, pressure: np.ndarray):
    """Interior-face Darcy fluxes ``T * (p_owner - p_neighbour)`` per direction.

    ``pressure`` is an ``(nx, ny, nz)`` field. Returns ``(fx, fy, fz)`` arrays on
    the interior x/y/z faces.
    """
    tx, ty, tz = _face_transmissibilities(grid)
    fx = tx * (pressure[:-1, :, :] - pressure[1:, :, :])
    fy = ty * (pressure[:, :-1, :] - pressure[:, 1:, :])
    fz = tz * (pressure[:, :, :-1] - pressure[:, :, 1:])
    return fx, fy, fz
