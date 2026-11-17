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


def test_permeability_field_reduces_to_scalar():
    from perfusion.core.darcy import Grid, transmissibilities

    g_scalar = Grid(nx=12, ny=9, lx=2.0, ly=1.5, permeability=3.0)
    g_field = Grid(nx=12, ny=9, lx=2.0, ly=1.5, permeability=np.full((12, 9), 3.0))
    txs, tys = transmissibilities(g_scalar)
    txf, tyf = transmissibilities(g_field)
    assert np.allclose(txs, txf)
    assert np.allclose(tys, tyf)


def test_transmissibility_is_harmonic_mean():
    from perfusion.core.darcy import Grid, transmissibilities

    k = np.ones((4, 3))
    k[2:, :] = 4.0                     # K = 1 (ix<2) | 4 (ix>=2); hx = hy = 1
    tx, _ = transmissibilities(Grid(nx=4, ny=3, lx=4.0, ly=3.0, permeability=k))
    assert tx[2, 0] == pytest.approx(2 * 1 * 4 / (1 + 4))   # harmonic mean = 1.6
    assert tx[1, 0] == pytest.approx(1.0)                   # K = 1 | 1 face


def test_low_permeability_barrier_concentrates_pressure_drop():
    # A low-K wall spanning the domain cannot reduce the (conserved) through-flux,
    # but it throttles conductance, so almost the entire pressure drop localises
    # across the wall — the signature of the harmonic-mean transmissibility.
    import scipy.sparse.linalg as spla

    from perfusion.core.darcy import Grid, tpfa_operator

    n = 24
    k = np.ones((n, n))
    k[n // 2, :] = 1e-6                # a near-impermeable vertical wall
    g = Grid(nx=n, ny=n, permeability=k)
    src = np.zeros(n * n)
    src[g.sub2ind(0, np.arange(n))] = 1.0
    src[g.sub2ind(n - 1, np.arange(n))] = -1.0
    a = tpfa_operator(g)[0].tolil()
    a[0, :] = 0.0
    a[0, 0] = 1.0
    src[0] = 0.0
    p = spla.spsolve(a.tocsr(), src).reshape(n, n, order="F")
    col = p.mean(axis=1)                                    # mean pressure per x-column
    drop_wall = abs(col[n // 2 - 1] - col[n // 2 + 1])      # across the wall
    drop_healthy = abs(col[n // 4] - col[n // 4 + 2])       # a healthy interior span
    assert drop_wall > 1e3 * drop_healthy                  # the barrier dominates the drop

def test_rasterize_permeability_lesion():
    from perfusion.core.geometry import rasterize_permeability

    dom = (0.0, 10.0, 0.0, 10.0)
    lesion = [([(3, 3), (7, 3), (7, 7), (3, 7)], 0.1)]   # square patch, K=0.1
    field = rasterize_permeability(20, 20, dom, lesion, base=1.0)
    assert field.shape == (20, 20)
    assert field.max() == pytest.approx(1.0)             # healthy tissue
    assert field.min() == pytest.approx(0.1)             # lesion
    assert (field < 1.0).sum() > 0
    assert rasterize_permeability(5, 5, dom, [], base=2.0).min() == pytest.approx(2.0)


def test_anisotropic_permeability_axis_split():
    from perfusion.core.darcy import Grid, transmissibilities

    # isotropic: permeability_y=None reduces to the scalar case
    iso = transmissibilities(Grid(nx=10, ny=8, permeability=2.0))
    same = transmissibilities(Grid(nx=10, ny=8, permeability=2.0, permeability_y=2.0))
    assert np.allclose(iso[0], same[0])
    assert np.allclose(iso[1], same[1])

    # Kx = 10, Ky = 1 on a square unit grid → interior tx is 10× ty
    g = Grid(nx=12, ny=12, permeability=10.0, permeability_y=1.0)
    tx, ty = transmissibilities(g)
    assert tx[6, 6] == pytest.approx(10.0 * ty[6, 6])


def test_anisotropic_throughflow_scales_with_aligned_axis():
    import scipy.sparse.linalg as spla

    from perfusion.core.darcy import Grid, face_fluxes, tpfa_operator

    def x_throughflow(kx):
        n = 20
        g = Grid(nx=n, ny=n, permeability=kx, permeability_y=1.0)
        a, tx, ty = tpfa_operator(g)
        a = a.tolil()
        b = np.zeros(n * n)
        for j in range(n):
            lo, hi = g.sub2ind(0, j), g.sub2ind(n - 1, j)
            a[lo, :] = 0
            a[lo, lo] = 1
            b[lo] = 1.0
            a[hi, :] = 0
            a[hi, hi] = 1
            b[hi] = 0.0
        p = spla.spsolve(a.tocsr(), b).reshape(n, n, order="F")
        fx, _ = face_fluxes(g, p, tx, ty)
        return np.abs(fx).sum()

    # a pure x-pressure ramp: throughflow is proportional to Kx
    assert x_throughflow(10.0) == pytest.approx(10.0 * x_throughflow(1.0), rel=1e-6)
