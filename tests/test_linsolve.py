"""AMG-preconditioned Darcy solve (optional ``amg`` extra)."""

import numpy as np
import pytest

from perfusion.core.darcy import Grid, tpfa_operator
from perfusion.core.linsolve import solve_spd


def test_direct_solver_matches_numpy():
    g = Grid(nx=20, ny=18)
    a = tpfa_operator(g)[0].tolil()
    a[0, :] = 0
    a[0, 0] = 1
    a = a.tocsr()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(a.shape[0])
    b[0] = 0.0
    x = solve_spd(a, b, method="direct")
    assert np.allclose(a @ x, b, atol=1e-8)


def test_amg_matches_direct_on_a_darcy_system():
    pytest.importorskip("pyamg")
    g = Grid(nx=40, ny=40)
    a = tpfa_operator(g)[0].tolil()
    a[0, :] = 0
    a[0, 0] = 1
    a = a.tocsr()
    rng = np.random.default_rng(1)
    b = rng.standard_normal((a.shape[0], 3))       # multi-RHS
    b[0, :] = 0.0
    direct = solve_spd(a, b, method="direct")
    amg = solve_spd(a, b, method="amg")
    assert np.allclose(direct, amg, atol=1e-6)


def test_amg_scales_the_3d_solve():
    pytest.importorskip("pyamg")
    from perfusion.core.darcy3d import Grid3D, solve_pressure_3d

    g = Grid3D(nx=12, ny=12, nz=12)
    src = np.zeros(g.n_cells)
    src[10], src[-10] = 1.0, -1.0
    assert np.allclose(solve_pressure_3d(g, src, solver="direct"),
                       solve_pressure_3d(g, src, solver="amg"), atol=1e-6)


def test_unknown_method_raises():
    g = Grid(nx=8, ny=8)
    a = tpfa_operator(g)[0].tocsr()
    with pytest.raises(ValueError, match="unknown method"):
        solve_spd(a, np.zeros(a.shape[0]), method="bogus")
