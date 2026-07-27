"""Darcy (porous-medium) model of the capillary bed with a TPFA discretisation.

The capillary bed is a structured 2D grid on which single-phase Darcy flow,
``-div(K/mu grad P) = q``, is discretised with a two-point flux approximation
(TPFA); see thesis Eqs. (4.16), (5.1)-(5.4). Point-source terminals from the
vascular network are regularised by a compactly supported mollifier
(Eq. 4.20 / 2.11).

Cells are numbered column-major (Fortran order) so ``linear = ix + iy * nx``,
matching the MATLAB ``reshape``/``sub2ind`` conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Grid:
    """Uniform structured grid for the Darcy capillary domain."""

    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0
    permeability: float | np.ndarray = 1.0   # isotropic K: scalar or a per-cell (nx, ny) field
    viscosity: float = 1.0
    porosity: float = 1.0
    thickness: float = 1.0          # out-of-plane thickness (2.5D)
    mask: np.ndarray | None = None  # True where the domain is active

    hx: float = field(init=False)
    hy: float = field(init=False)

    def __post_init__(self) -> None:
        self.hx = self.lx / self.nx
        self.hy = self.ly / self.ny
        if self.mask is None:
            self.mask = np.ones((self.nx, self.ny), dtype=bool)
        if np.ndim(self.permeability) != 0:
            k = np.asarray(self.permeability, dtype=float)
            if k.shape != (self.nx, self.ny):
                raise ValueError(
                    f"permeability field {k.shape} must be (nx, ny) = {(self.nx, self.ny)}")
            self.permeability = k

    def permeability_field(self) -> np.ndarray:
        """Per-cell permeability as an ``(nx, ny)`` array (broadcasts a scalar K)."""
        if np.ndim(self.permeability) == 0:
            return np.full((self.nx, self.ny), float(self.permeability))
        return self.permeability

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_volume(self) -> float:
        return self.hx * self.hy * self.thickness

    def sub2ind(self, ix, iy) -> np.ndarray:
        """Column-major linear index (matches MATLAB ``sub2ind([nx ny], ...)``)."""
        return np.asarray(ix) + np.asarray(iy) * self.nx

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Flattened (column-major) cell-centre coordinates ``(xc, yc)``."""
        xc = (np.arange(self.nx) + 0.5) * self.hx
        yc = (np.arange(self.ny) + 0.5) * self.hy
        gx, gy = np.meshgrid(xc, yc, indexing="ij")
        return gx.ravel(order="F"), gy.ravel(order="F")


def face_conductance(grid: Grid) -> float:
    """Constant interface conductance ``K/mu`` for the *homogeneous* case.

    Retained as a convenience for scalar K; heterogeneous grids use the per-face
    harmonic mean assembled directly in :func:`transmissibilities`.
    """
    if np.ndim(grid.permeability) != 0:
        raise ValueError("face_conductance is scalar-only; use transmissibilities()")
    return grid.permeability / grid.viscosity


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Harmonic mean ``2ab/(a+b)`` of adjacent-cell permeabilities (0 if both 0)."""
    denom = a + b
    return np.divide(2.0 * a * b, denom, out=np.zeros_like(denom), where=denom > 0.0)


def transmissibilities(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Interface transmissibilities ``(Tx, Ty)`` with no-flow outer boundaries.

    Shapes ``(nx+1, ny)`` and ``(nx, ny+1)``; only interior faces are non-zero.
    The interface permeability is the **harmonic mean** of the two adjacent cells'
    permeabilities (Eq. 5.1) — exact for a uniform grid and the correct averaging
    for a heterogeneous ``K`` field (a low-K cell throttles both its faces). Faces
    touching an inactive (masked-out) cell carry no flow, so the active domain is
    hydraulically isolated from the surroundings.
    """
    kf = grid.permeability_field() / grid.viscosity
    tx = np.zeros((grid.nx + 1, grid.ny))
    ty = np.zeros((grid.nx, grid.ny + 1))
    tx[1:grid.nx, :] = _harmonic(kf[:-1, :], kf[1:, :]) * grid.hy * grid.thickness / grid.hx
    ty[:, 1:grid.ny] = _harmonic(kf[:, :-1], kf[:, 1:]) * grid.hx * grid.thickness / grid.hy
    # a face conducts only when both adjacent cells are active
    active = grid.mask
    tx[1:grid.nx, :] *= active[:-1, :] & active[1:, :]
    ty[:, 1:grid.ny] *= active[:, :-1] & active[:, 1:]
    return tx, ty


def internal_faces(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Owner/neighbour cell indices for every internal face.

    Returns ``(owner, neighbour)`` 0-based linear indices. A positive face flux
    (see :func:`face_flux_vector`) points from ``owner`` to ``neighbour``. The
    ordering is x-faces then y-faces, matching ``face_flux_vector``.
    """
    nx, ny = grid.nx, grid.ny
    cell = np.arange(grid.n_cells).reshape(nx, ny, order="F")
    # x-faces between (ix-1, iy) and (ix, iy), ix = 1..nx-1
    xo = cell[:-1, :].ravel(order="F")
    xn = cell[1:, :].ravel(order="F")
    # y-faces between (ix, iy-1) and (ix, iy), iy = 1..ny-1
    yo = cell[:, :-1].ravel(order="F")
    yn = cell[:, 1:].ravel(order="F")
    owner = np.concatenate([xo, yo])
    neighbour = np.concatenate([xn, yn])
    return owner, neighbour


def face_flux_vector(grid: Grid, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Flatten interior face fluxes to align with :func:`internal_faces`."""
    fxi = fx[1:grid.nx, :].ravel(order="F")   # interior x-faces
    fyi = fy[:, 1:grid.ny].ravel(order="F")   # interior y-faces
    return np.concatenate([fxi, fyi])


def tpfa_operator(grid: Grid) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the TPFA matrix ``A`` for ``-div(K/mu grad P) = q``.

    Returns ``(A, Tx, Ty)``. ``A`` is singular (pure Neumann): pin a cell or add
    a balanced source/sink before solving.
    """
    nx, ny, n = grid.nx, grid.ny, grid.n_cells
    tx, ty = transmissibilities(grid)

    west = tx[0:nx, :].ravel(order="F")
    east = tx[1:nx + 1, :].ravel(order="F")
    south = ty[:, 0:ny].ravel(order="F")
    north = ty[:, 1:ny + 1].ravel(order="F")
    diag = west + east + south + north

    A = sp.diags(
        [-south[nx:], -west[1:], diag, -east[:-1], -north[:-nx]],
        [nx, 1, 0, -1, -nx],
        shape=(n, n), format="csr",
    )
    # Inactive cells are decoupled (all their faces are no-flow); give them an
    # identity row so the operator is non-singular there (p = 0).
    inactive = ~grid.mask.ravel(order="F")
    if inactive.any():
        A = A + sp.diags(inactive.astype(float), format="csr")
    return A, tx, ty


def mollifier(grid: Grid, center: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Compactly supported source distribution around a terminal cell (Eq. 4.20).

    Uses the thesis bump kernel ``exp(-1/(1-r^2))`` (Eq. 4.20 / 2.11), which is
    far less prone to floating-point underflow than the steeper ``^8`` variant.
    Returns ``(cells, weights)`` with weights summing to 1 over the support of
    radius ``eps`` (in cell units), restricted to the active domain mask.
    """
    xc, yc = grid.cell_centers()
    dist2 = ((xc - xc[center]) / grid.hx) ** 2 + ((yc - yc[center]) / grid.hy) ** 2
    support = (dist2 <= eps**2) & grid.mask.ravel(order="F")
    if not support.any():
        support[center] = True   # degenerate: keep the terminal cell itself
    normalised = np.clip(dist2[support] / eps**2, 0.0, 1.0 - 1e-12)
    weights = np.exp(-1.0 / (1.0 - normalised))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones_like(weights)   # degenerate support; fall back to uniform
        total = weights.sum()
    weights /= total
    return np.flatnonzero(support), weights


def terminal_response(grid: Grid, terminal_cells: np.ndarray, eps: float,
                      pin_cell: int | None = None):
    """Darcy pressure response of the capillary bed to unit terminal fluxes.

    Solves the (Dirichlet-pinned) TPFA system for every mollified terminal source
    in a single batched back-substitution, and returns:

    - ``response`` : ``(T, T)`` terminal-to-terminal pressure response;
      ``response[i, j]`` is the pressure at terminal ``i`` per unit flux injected
      at terminal ``j`` (the block eliminated by the Schur complement).
    - ``field_response`` : ``(n_cells, T)`` full-field pressure response columns.
    - ``sources`` : list of ``(cells, weights)`` per terminal (reused by transport).
    """
    n_terminals = len(terminal_cells)
    if pin_cell is None:
        pin_cell = int(np.flatnonzero(grid.mask.ravel(order="F"))[0])  # first active cell
    A = tpfa_operator(grid)[0].tolil()
    A[pin_cell, :] = 0.0
    A[pin_cell, pin_cell] = 1.0            # Dirichlet pin removes the nullspace
    lu = spla.splu(A.tocsc())

    sources = [mollifier(grid, int(c), eps) for c in terminal_cells]
    rhs = np.zeros((grid.n_cells, n_terminals))
    for j, (cells, weights) in enumerate(sources):
        rhs[cells, j] = weights
    rhs[pin_cell, :] = 0.0
    field_response = lu.solve(rhs) if n_terminals else np.zeros((grid.n_cells, 0))

    response = np.zeros((n_terminals, n_terminals))
    for i, (cells, weights) in enumerate(sources):
        response[i, :] = weights @ field_response[cells, :]
    return response, field_response, sources


def reconstruct_field(field_response: np.ndarray, terminal_flux: np.ndarray,
                      balance: bool = True) -> np.ndarray:
    """Superpose per-terminal Darcy responses into a full pressure field.

    Parameters
    ----------
    field_response : ``(n_cells, T)`` response columns from
        :func:`terminal_response`.
    terminal_flux : ``(T,)`` signed terminal fluxes (+ arterial in, - venous out).
    balance : rescale arterial inflow to match venous outflow so the pinned cell
        carries no net flux (mirrors ``Frog.m``).

    Returns
    -------
    ``(n_cells,)`` flattened pressure field (column-major).
    """
    flux = np.array(terminal_flux, dtype=float)
    if balance:
        inflow = flux[flux > 0].sum()
        outflow = -flux[flux < 0].sum()
        if inflow > 0 and outflow > 0:
            flux = flux.copy()
            flux[flux > 0] *= outflow / inflow
    return field_response @ flux


def face_fluxes(grid: Grid, pressure: np.ndarray, tx: np.ndarray, ty: np.ndarray):
    """Interface volumetric fluxes from a cell-centred pressure field ``(nx, ny)``.

    Returns staggered ``(Fx, Fy)`` of shape ``(nx+1, ny)`` and ``(nx, ny+1)``.
    """
    nx, ny = grid.nx, grid.ny
    fx = np.zeros((nx + 1, ny))
    fy = np.zeros((nx, ny + 1))
    fx[1:nx, :] = (pressure[0:nx - 1, :] - pressure[1:nx, :]) * tx[1:nx, :]
    fy[:, 1:ny] = (pressure[:, 0:ny - 1] - pressure[:, 1:ny]) * ty[:, 1:ny]
    return fx, fy


def upwind_operator(grid: Grid, fx: np.ndarray, fy: np.ndarray):
    """Sparse first-order upwind advection operator for the internal faces.

    Returns ``(M, outflow)`` where, for a cell-centred field ``c``,
    ``(M @ c)[i]`` is the net advective inflow into cell ``i`` from its
    neighbours and ``outflow[i]`` is its total outward face flux. Both the
    time-of-flight and tracer-transport solvers are built from this.
    """
    n = grid.n_cells
    owner, neighbour = internal_faces(grid)
    flux = face_flux_vector(grid, fx, fy)
    to_neighbour = np.maximum(flux, 0.0)   # owner -> neighbour
    to_owner = -np.minimum(flux, 0.0)      # neighbour -> owner

    # inflow to `neighbour` from `owner`, and to `owner` from `neighbour`
    M = (sp.coo_matrix((to_neighbour, (neighbour, owner)), shape=(n, n))
         + sp.coo_matrix((to_owner, (owner, neighbour)), shape=(n, n)))
    outflow = np.bincount(np.concatenate([owner, neighbour]),
                          weights=np.concatenate([to_neighbour, to_owner]),
                          minlength=n)
    return M.tocsr(), outflow
