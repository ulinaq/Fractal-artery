"""Tests for the TPFA Darcy capillary-bed model."""

import numpy as np
import pytest

from perfusion import darcy


def test_grid_geometry_and_indexing(small_grid):
    g = small_grid
    assert g.n_cells == g.nx * g.ny
    assert g.cell_volume == pytest.approx(g.hx * g.hy * g.thickness)
    # column-major: cell (ix, iy) -> ix + iy*nx
    assert g.sub2ind(2, 3) == 2 + 3 * g.nx
    xc, _ = g.cell_centers()
    assert xc.shape == (g.n_cells,)
    assert xc[0] == pytest.approx(0.5 * g.hx)


def test_transmissibilities_interior_only(small_grid):
    tx, ty = darcy.transmissibilities(small_grid)
    assert tx.shape == (small_grid.nx + 1, small_grid.ny)
    assert ty.shape == (small_grid.nx, small_grid.ny + 1)
    # outer faces carry no flow
    assert np.all(tx[0, :] == 0)
    assert np.all(tx[-1, :] == 0)
    assert np.all(ty[:, 0] == 0)
    assert np.all(ty[:, -1] == 0)
    assert np.all(tx[1:-1, :] > 0)


def test_tpfa_rowsum_zero_interior(small_grid):
    # Pure Neumann operator: interior rows sum to zero (conservative stencil).
    A, _, _ = darcy.tpfa_operator(small_grid)
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 0.0, atol=1e-12)


def test_tpfa_linear_pressure_ramp(small_grid):
    # Dirichlet gradient across x reproduces a linear pressure profile.
    g = small_grid
    A = darcy.tpfa_operator(g)[0].tolil()
    rhs = np.zeros(g.n_cells)
    for iy in range(g.ny):
        left, right = g.sub2ind(0, iy), g.sub2ind(g.nx - 1, iy)
        A[left, :] = 0
        A[left, left] = 1
        rhs[left] = 1.0
        A[right, :] = 0
        A[right, right] = 1
        rhs[right] = 0.0
    p = np.linalg.solve(A.toarray(), rhs).reshape(g.nx, g.ny, order="F")
    assert np.allclose(p[:, g.ny // 2], np.linspace(1, 0, g.nx), atol=0.05)


def test_mollifier_partition_of_unity(small_grid):
    cells, weights = darcy.mollifier(small_grid, center=small_grid.sub2ind(20, 20),
                                     eps=4.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)
    assert cells.size > 1


def test_mollifier_masked_no_nan(masked_grid):
    terminals = masked_grid.sub2ind([10, 18], [10, 18])
    response, field_response, sources = darcy.terminal_response(
        masked_grid, terminals, eps=4.0)
    assert np.isfinite(response).all()
    assert np.isfinite(field_response).all()
    assert len(sources) == 2


def test_mask_isolates_domain(masked_grid):
    # Faces touching an inactive cell carry no flow, and inactive cells decouple.
    tx, _ = darcy.transmissibilities(masked_grid)
    active = masked_grid.mask
    # any x-face with an inactive neighbour must be zero
    face_active = active[:-1, :] & active[1:, :]
    assert np.all(tx[1:masked_grid.nx, :][~face_active] == 0)
    # reconstructed pressure is exactly zero on inactive cells
    terminals = masked_grid.sub2ind([10, 18], [10, 18])
    _, field_response, _ = darcy.terminal_response(masked_grid, terminals, eps=3.0)
    field = darcy.reconstruct_field(field_response, np.array([1.0, -1.0]))
    inactive = ~masked_grid.mask.ravel(order="F")
    assert np.allclose(field[inactive], 0.0)


def test_mollifier_kernel_is_soft(small_grid):
    # The thesis power-1 bump keeps essentially the whole support positive
    # (the ^8 kernel underflowed most interior cells to exactly zero).
    _, weights = darcy.mollifier(small_grid, small_grid.sub2ind(20, 20), eps=4.0)
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0)
    assert np.mean(weights > 0) > 0.9


def test_terminal_response_shapes(small_grid):
    terminals = small_grid.sub2ind([10, 30], [10, 30])
    response, field_response, _ = darcy.terminal_response(
        small_grid, terminals, eps=3.0)
    assert response.shape == (2, 2)
    assert field_response.shape == (small_grid.n_cells, 2)
    # response is roughly symmetric (reciprocity of the pinned Green's function)
    assert response == pytest.approx(response.T, abs=1e-6)


def test_reconstruct_field_balances_flux():
    field_response = np.array([[1.0, -1.0], [2.0, -0.5]])
    # unbalanced fluxes get rescaled so inflow == outflow
    out = darcy.reconstruct_field(field_response, np.array([4.0, -2.0]), balance=True)
    # arterial inflow 4 scaled to match outflow 2 -> flux = [2, -2]
    assert np.allclose(out, field_response @ np.array([2.0, -2.0]))


def test_internal_faces_and_flux_vector(small_grid):
    owner, neighbour = darcy.internal_faces(small_grid)
    n_internal = (small_grid.nx - 1) * small_grid.ny + small_grid.nx * (small_grid.ny - 1)
    assert owner.shape == (n_internal,)
    assert np.all(owner != neighbour)
    # a uniform pressure field yields zero face fluxes
    tx, ty = darcy.transmissibilities(small_grid)
    fx, fy = darcy.face_fluxes(small_grid, np.ones((small_grid.nx, small_grid.ny)),
                               tx, ty)
    assert np.allclose(darcy.face_flux_vector(small_grid, fx, fy), 0.0)


def test_upwind_operator_conserves(small_grid):
    # Build a simple left-to-right flow field and check the operator structure.
    g = small_grid
    tx, ty = darcy.transmissibilities(g)
    p = np.tile(np.linspace(1, 0, g.nx)[:, None], (1, g.ny))
    fx, fy = darcy.face_fluxes(g, p, tx, ty)
    M, outflow = darcy.upwind_operator(g, fx, fy)
    inflow = np.asarray(M.sum(axis=1)).ravel()
    # interior cells: inflow balances outflow for a divergence-free field
    interior = np.zeros((g.nx, g.ny), dtype=bool)
    interior[1:-1, 1:-1] = True
    interior = interior.ravel(order="F")
    assert np.allclose(inflow[interior], outflow[interior], atol=1e-9)
