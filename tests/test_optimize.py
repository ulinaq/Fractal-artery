"""Tests for the vascular-structure optimisation objectives (Paper D)."""

import numpy as np
import pytest

from perfusion import Grid, optimize


def test_default_alpha_layout():
    a = optimize.default_alpha()
    assert a.shape == (optimize.ALPHA_LEN,)
    assert a[8] == pytest.approx(2.8)             # bifurcation exponent


def test_params_from_alpha_roundtrip():
    a = optimize.default_alpha()
    prm = optimize.params_from_alpha(a)
    assert prm.length_to_radius == pytest.approx(a[2])
    assert prm.exponent == pytest.approx(a[8])
    assert prm.a_gamma == pytest.approx(a[0])
    # deeper-generation tables are populated deterministically from alpha
    assert prm.phi.shape[0] == 5
    assert prm.vphi.shape[0] == 8


@pytest.fixture(scope="module")
def coarse_grid():
    return Grid(nx=70, ny=80, lx=140, ly=160)


@pytest.mark.parametrize("kind", ["flux_std", "flux_range", "flux_l2",
                                  "flux_mean", "tof_std", "tof_max"])
def test_objectives_finite_and_positive(coarse_grid, kind):
    val = optimize.objective(optimize.default_alpha(), kind=kind, grid=coarse_grid)
    assert np.isfinite(val)
    assert val > 0


def test_invalid_geometry_returns_inf(coarse_grid):
    # Push the arterial root far outside the domain; strict mode rejects it.
    a = optimize.default_alpha()
    a[6] = 1e6
    assert optimize.objective(a, kind="flux_std", grid=coarse_grid,
                              strict=True) == float("inf")


def test_unknown_objective_raises(coarse_grid):
    with pytest.raises(ValueError, match="unknown objective"):
        optimize.objective(optimize.default_alpha(), kind="bogus", grid=coarse_grid)


def test_optimize_tree_local_improves(coarse_grid):
    from perfusion import Parameters

    start = optimize.objective(optimize.default_alpha(), kind="flux_std", grid=coarse_grid)
    r = optimize.optimize_tree(kind="flux_std", grid=coarse_grid, method="local", maxiter=8)
    assert len(r.alpha) == optimize.ALPHA_LEN
    assert isinstance(r.params, Parameters)
    assert r.n_eval > 0
    assert r.objective <= start + 1e-9        # a descent method never worsens from the start
    assert r.free == [0, 1]                    # default: the two branching asymmetries


def test_optimize_tree_unknown_method(coarse_grid):
    with pytest.raises(ValueError, match="unknown method"):
        optimize.optimize_tree(grid=coarse_grid, method="bogus", maxiter=1)
