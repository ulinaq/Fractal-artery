"""Monte-Carlo uncertainty quantification + sensitivity (perfusion.uq)."""

import numpy as np
import pytest

from perfusion import Grid, uq


@pytest.fixture(scope="module")
def grid():
    return Grid(nx=45, ny=52, lx=140, ly=160)


def test_latin_hypercube_is_stratified():
    rng = np.random.default_rng(0)
    s = uq.latin_hypercube(20, 3, rng)
    assert s.shape == (20, 3)
    assert s.min() >= 0.0
    assert s.max() < 1.0
    # exactly one sample per 1/n stratum along each axis
    for j in range(3):
        strata = np.floor(s[:, j] * 20).astype(int)
        assert sorted(strata) == list(range(20))


def test_sensitivity_signs_are_physical(grid):
    res = uq.monte_carlo(
        {"p_in": (5.0, 20.0), "mu": (0.8, 1.5), "r_min_frac": (0.35, 0.6)},
        grid, n_samples=24, base={"p_out": 0.0}, seed=1)
    assert res.qoi_names == ["arterial_inflow", "flux_mean", "tof_mean"]
    s = res.sensitivity["arterial_inflow"]
    assert s["p_in"] > 0.7          # higher inlet pressure → more inflow
    assert s["mu"] < 0.0            # higher viscosity → less inflow
    # transit time moves opposite to inflow w.r.t. inlet pressure
    assert res.sensitivity["tof_mean"]["p_in"] < 0.0
    # every QoI actually varied, and mean sits inside the 5–95 envelope
    for q in res.qoi_names:
        assert res.std[q] > 0.0
        assert res.p05[q] <= res.mean[q] <= res.p95[q]


def test_permeability_as_an_uncertain_input(grid):
    res = uq.monte_carlo({"permeability": (0.5, 2.0)}, grid, n_samples=12, seed=2)
    assert res.names == ["permeability"]
    assert np.all(np.isfinite(res.outputs))


def test_determinism_and_validation(grid):
    a = uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=10, seed=3)
    b = uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=10, seed=3)
    assert np.allclose(a.outputs, b.outputs)
    with pytest.raises(ValueError, match="n_samples"):
        uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=1)
    with pytest.raises(ValueError, match="unknown uncertain parameter"):
        uq.monte_carlo({"bogus": (0.0, 1.0)}, grid, n_samples=4)
