"""Tests for the 3-D structured Darcy grid + TPFA operator (Phase 4 foundation)."""

import numpy as np
import scipy.sparse.linalg as spla

from perfusion.core.darcy3d import (
    Grid3D,
    face_fluxes_3d,
    point_source_3d,
    solve_pressure_3d,
    tpfa_operator_3d,
)


def _solve_x_ramp(grid, p_lo=1.0, p_hi=0.0):
    """Dirichlet p_hi on the x=0 layer, p_lo on the x=nx-1 layer; solve interior."""
    A = tpfa_operator_3d(grid).tolil()
    b = np.zeros(grid.n_cells)
    cell = np.arange(grid.n_cells).reshape(grid.nx, grid.ny, grid.nz, order="F")
    lo, hi = cell[0, :, :].ravel(), cell[-1, :, :].ravel()
    for face, value in ((hi, p_hi), (lo, p_lo)):
        for c in face:
            A.rows[c], A.data[c] = [c], [1.0]        # identity row
            b[c] = value
    p = spla.spsolve(A.tocsr(), b)
    return p.reshape(grid.nx, grid.ny, grid.nz, order="F")


def test_operator_is_square_and_singular():
    grid = Grid3D(nx=5, ny=4, nz=3)
    A = tpfa_operator_3d(grid)
    assert A.shape == (grid.n_cells, grid.n_cells)
    # Neumann operator: constant pressure is a null vector (row sums ~ 0)
    assert np.allclose(np.asarray(A.sum(axis=1)).ravel(), 0.0, atol=1e-12)


def test_linear_pressure_ramp():
    grid = Grid3D(nx=10, ny=5, nz=4, lx=10, ly=5, lz=4)
    p = _solve_x_ramp(grid, p_lo=1.0, p_hi=0.0)
    # Dirichlet is imposed at the first/last cell-centre layers, so the TPFA
    # solution is linear in the layer index: 1 at layer 0 -> 0 at layer nx-1.
    expected = 1.0 - np.arange(grid.nx) / (grid.nx - 1)
    # pressure depends only on x and is linear; independent of y, z
    for iz in range(grid.nz):
        for iy in range(grid.ny):
            assert np.allclose(p[:, iy, iz], expected, atol=1e-10)


def test_ramp_flux_is_uniform_in_x_and_zero_transverse():
    grid = Grid3D(nx=8, ny=5, nz=4, lx=8, ly=5, lz=4)
    p = _solve_x_ramp(grid, p_lo=1.0, p_hi=0.0)
    fx, fy, fz = face_fluxes_3d(grid, p)
    assert np.allclose(fx, fx.flat[0])                   # same flux through every x-face
    assert fx.flat[0] > 0                                # flows high -> low pressure
    assert np.allclose(fy, 0.0, atol=1e-10)              # no transverse flow
    assert np.allclose(fz, 0.0, atol=1e-10)


def test_mask_isolates_inactive_cells():
    grid = Grid3D(nx=6, ny=6, nz=6)
    grid.mask[:, :, 0] = False                            # deactivate a z-slab
    A = tpfa_operator_3d(grid)
    inactive = ~grid.mask.ravel(order="F")
    rows = A.tocsr()[inactive]
    # inactive cells are identity rows (isolated): one nonzero of 1.0 each
    assert rows.nnz == int(inactive.sum())
    assert np.allclose(rows.data, 1.0)


def test_point_source_is_normalised_and_local():
    grid = Grid3D(nx=9, ny=9, nz=9)
    center = int(grid.sub2ind(4, 4, 4))
    cells, weights = point_source_3d(grid, center, eps=2.0)
    assert np.isclose(weights.sum(), 1.0)          # weights sum to 1
    assert center in cells.tolist()                # includes the centre
    assert cells.size < grid.n_cells               # compact support


def test_source_sink_darcy_solve():
    grid = Grid3D(nx=12, ny=10, nz=8, lx=12, ly=10, lz=8)
    a = int(grid.sub2ind(2, 5, 4))                 # injection cell
    b = int(grid.sub2ind(9, 5, 4))                 # extraction cell
    q = np.zeros(grid.n_cells)
    ca, wa = point_source_3d(grid, a, eps=1.5)
    cb, wb = point_source_3d(grid, b, eps=1.5)
    q[ca] += wa                                     # +1 injected
    q[cb] -= wb                                     # -1 extracted (balanced)
    assert np.isclose(q.sum(), 0.0, atol=1e-12)

    p = solve_pressure_3d(grid, q).ravel(order="F")
    # conservation holds everywhere except the pinned datum cell
    A = tpfa_operator_3d(grid)
    pin = 0
    residual = (A @ p) - q
    residual[pin] = 0.0
    assert np.allclose(residual, 0.0, atol=1e-8)
    # pressure is higher at the source than the sink -> flow a -> b
    assert p[a] > p[b]
