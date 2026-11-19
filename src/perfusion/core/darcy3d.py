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
    permeability: float | np.ndarray = 1.0   # scalar K, or a per-voxel (nx, ny, nz) field
    thickness: float = 1.0                 # unused in 3-D; kept for API symmetry
    porosity: float = 1.0                  # pore fraction (time-of-flight pore volume)
    mask: np.ndarray | None = None         # (nx, ny, nz) bool; True = active

    hx: float = field(init=False)
    hy: float = field(init=False)
    hz: float = field(init=False)

    def __post_init__(self) -> None:
        if min(int(self.nx), int(self.ny), int(self.nz)) < 1:
            raise ValueError(f"grid needs nx, ny, nz >= 1 (got {self.nx}x{self.ny}x{self.nz})")
        self.hx = self.lx / self.nx
        self.hy = self.ly / self.ny
        self.hz = self.lz / self.nz
        if self.mask is None:
            self.mask = np.ones((self.nx, self.ny, self.nz), dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != (self.nx, self.ny, self.nz):
                raise ValueError(f"mask {self.mask.shape} must be (nx, ny, nz) = "
                                 f"{(self.nx, self.ny, self.nz)}")
        if np.ndim(self.permeability) and np.shape(self.permeability) != (
                self.nx, self.ny, self.nz):
            raise ValueError(f"permeability field {np.shape(self.permeability)} must be "
                             f"({self.nx}, {self.ny}, {self.nz})")

    def permeability_field(self) -> np.ndarray:
        """Per-voxel permeability ``(nx, ny, nz)`` (broadcasts a scalar K)."""
        if np.ndim(self.permeability) == 0:
            return np.full((self.nx, self.ny, self.nz), float(self.permeability))
        return np.asarray(self.permeability, dtype=float)

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
    """Interior-face transmissibilities ``(tx, ty, tz)``, shapes ``(nx-1, ny, nz)``,
    ``(nx, ny-1, nz)``, ``(nx, ny, nz-1)``. The face permeability is the **harmonic
    mean** of its two voxels (as in 2-D, Eq. 5.1) — exact for a uniform K, and a
    low-K voxel throttles all six of its faces."""
    from .darcy import _harmonic

    k = grid.permeability_field()
    tx = _harmonic(k[:-1, :, :], k[1:, :, :]) * grid.hy * grid.hz / grid.hx
    ty = _harmonic(k[:, :-1, :], k[:, 1:, :]) * grid.hx * grid.hz / grid.hy
    tz = _harmonic(k[:, :, :-1], k[:, :, 1:]) * grid.hx * grid.hy / grid.hz
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
        o, m, t = o[keep], m[keep], t.ravel(order="F")[keep]
        rows.extend((o, m))
        cols.extend((m, o))
        vals.extend((-t, -t))
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


def solve_pressure_3d(grid: Grid3D, source: np.ndarray, *, solver: str = "direct") -> np.ndarray:
    """Solve ``-div(K grad p) = source`` on the active 3-D domain.

    ``source`` is the net volumetric injection per cell (must sum to ~0 over the
    active domain for a well-posed Neumann problem). One active cell is pinned as
    the pressure datum. ``solver='amg'`` uses an algebraic-multigrid preconditioned
    CG (optional ``pyamg`` extra) which scales far better on large grids than the
    default direct LU. Returns the ``(nx, ny, nz)`` pressure field.
    """
    from .linsolve import solve_spd

    active = grid.mask.ravel(order="F")
    from .darcy import pin_row

    b = np.asarray(source, dtype=float).copy()
    pin = int(np.flatnonzero(active)[0])
    b[pin] = 0.0                                              # datum: p_pin = 0
    p = solve_spd(pin_row(tpfa_operator_3d(grid), pin).tocsr(), b, method=solver)
    return p.reshape(grid.nx, grid.ny, grid.nz, order="F")


def terminal_response_3d(grid: Grid3D, terminal_cells, eps: float, pin_cell=None):
    """3-D Darcy terminal response (Schur coupling), mirroring ``darcy.terminal_response``.

    Returns ``(response, field_response, sources)``: the ``(T, T)``
    terminal-to-terminal pressure response, the ``(n_cells, T)`` full-field
    response columns, and the per-terminal mollified sources.
    """
    n_terminals = len(terminal_cells)
    active = grid.mask.ravel(order="F")
    if pin_cell is None:
        pin_cell = int(np.flatnonzero(active)[0])
    from .darcy import pin_row

    lu = spla.splu(pin_row(tpfa_operator_3d(grid), pin_cell))   # Dirichlet pin

    sources = [point_source_3d(grid, int(c), eps) for c in terminal_cells]
    rhs = np.zeros((grid.n_cells, n_terminals))
    for j, (cells, weights) in enumerate(sources):
        rhs[cells, j] = weights
    rhs[pin_cell, :] = 0.0
    field_response = lu.solve(rhs) if n_terminals else np.zeros((grid.n_cells, 0))

    response = np.zeros((n_terminals, n_terminals))
    for i, (cells, weights) in enumerate(sources):
        response[i, :] = weights @ field_response[cells, :]
    return response, field_response, sources


def coupling_operator_3d(grid: Grid3D, terminal_cells, eps: float, pin_cell=None):
    """3-D twin of :func:`perfusion.core.darcy.coupling_operator`: the sparse
    monolithic coupling ``(MonolithicDarcy, sources)`` — no dense ``(n_cells, T)``
    field-response block, so fine 3-D grids with many terminals fit in memory."""
    from .darcy import _weight_matrices, pin_row
    from .network import MonolithicDarcy

    if pin_cell is None:
        pin_cell = int(np.flatnonzero(grid.mask.ravel(order="F"))[0])
    A = pin_row(tpfa_operator_3d(grid), pin_cell).tocsr()
    sources = [point_source_3d(grid, int(c), eps) for c in terminal_cells]
    return MonolithicDarcy(A, *_weight_matrices(grid.n_cells, sources, pin_cell)), sources


def face_fluxes_3d(grid: Grid3D, pressure: np.ndarray):
    """Interior-face Darcy fluxes ``T * (p_owner - p_neighbour)`` per direction.

    ``pressure`` is an ``(nx, ny, nz)`` field. Returns ``(fx, fy, fz)`` arrays on
    the interior x/y/z faces, shapes ``(nx-1, ny, nz)``, ``(nx, ny-1, nz)``,
    ``(nx, ny, nz-1)``. Faces touching an inactive cell carry no flow (as in the
    TPFA operator), so nothing leaks across an organ-shaped mask.
    """
    tx, ty, tz = _face_transmissibilities(grid)
    m = grid.mask
    p = np.where(m, pressure, 0.0)
    fx = tx * (p[:-1, :, :] - p[1:, :, :]) * (m[:-1, :, :] & m[1:, :, :])
    fy = ty * (p[:, :-1, :] - p[:, 1:, :]) * (m[:, :-1, :] & m[:, 1:, :])
    fz = tz * (p[:, :, :-1] - p[:, :, 1:]) * (m[:, :, :-1] & m[:, :, 1:])
    return fx, fy, fz


def flux_magnitude_3d(grid: Grid3D, fx, fy, fz) -> np.ndarray:
    """Cell-centred Darcy flux magnitude ``|v|`` ``(nx, ny, nz)`` from interior face
    fluxes (boundary faces carry none) — the 3-D twin of ``flux_magnitude``."""
    def centred(f, axis):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        g = np.pad(f, pad)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        return 0.5 * (g[tuple(lo)] + g[tuple(hi)])

    q = np.sqrt(centred(fx, 0) ** 2 + centred(fy, 1) ** 2 + centred(fz, 2) ** 2)
    return np.where(grid.mask, q, np.nan)       # same units as the 2-D flux_magnitude


def upwind_operator_3d(grid: Grid3D, fx, fy, fz):
    """Sparse first-order upwind advection operator over the interior faces —
    the 3-D twin of ``darcy.upwind_operator``. Returns ``(M, outflow)`` with
    ``(M @ c)[i]`` the net advective inflow into cell ``i``."""
    n = grid.n_cells
    cell = np.arange(n).reshape(grid.nx, grid.ny, grid.nz, order="F")
    pairs = [(cell[:-1, :, :], cell[1:, :, :], fx), (cell[:, :-1, :], cell[:, 1:, :], fy),
             (cell[:, :, :-1], cell[:, :, 1:], fz)]
    owner = np.concatenate([o.ravel(order="F") for o, _, _ in pairs])
    neigh = np.concatenate([m.ravel(order="F") for _, m, _ in pairs])
    flux = np.concatenate([f.ravel(order="F") for _, _, f in pairs])
    to_n, to_o = np.maximum(flux, 0.0), -np.minimum(flux, 0.0)
    M = (sp.coo_matrix((to_n, (neigh, owner)), shape=(n, n))
         + sp.coo_matrix((to_o, (owner, neigh)), shape=(n, n)))
    outflow = np.bincount(np.concatenate([owner, neigh]),
                          weights=np.concatenate([to_n, to_o]), minlength=n)
    return M.tocsr(), outflow
