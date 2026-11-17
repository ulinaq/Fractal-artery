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


def test_sobol_indices_are_well_behaved(grid):
    res = uq.sobol_indices(
        {"p_in": (5.0, 20.0), "mu": (0.8, 1.5), "r_min_frac": (0.35, 0.6)},
        grid, n_samples=48, base={"p_out": 0.0}, seed=1)
    s1 = res.first_order["arterial_inflow"]
    st = res.total_effect["arterial_inflow"]
    # the degree-2 surrogate should explain almost all of a smooth model's variance
    assert res.fit_r2["arterial_inflow"] > 0.9
    assert s1["p_in"] == max(s1.values())          # inlet pressure dominates
    for p in res.names:                            # total effect ≥ first order; both in [0,1]
        assert st[p] >= s1[p] - 1e-9
        assert 0.0 <= s1[p] <= 1.0001
        assert 0.0 <= st[p] <= 1.0001
    assert sum(s1.values()) <= 1.0001              # first-order shares can't exceed the whole


def test_sobol_is_deterministic_and_validates(grid):
    a = uq.sobol_indices({"p_in": (5.0, 20.0), "mu": (0.8, 1.5)}, grid, n_samples=24, seed=5)
    b = uq.sobol_indices({"p_in": (5.0, 20.0), "mu": (0.8, 1.5)}, grid, n_samples=24, seed=5)
    assert a.first_order == b.first_order
    with pytest.raises(ValueError, match="n_samples must be"):
        uq.sobol_indices({"p_in": (5.0, 20.0)}, grid, n_samples=4)


def test_determinism_and_validation(grid):
    a = uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=10, seed=3)
    b = uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=10, seed=3)
    assert np.allclose(a.outputs, b.outputs)
    with pytest.raises(ValueError, match="n_samples"):
        uq.monte_carlo({"p_in": (5.0, 20.0)}, grid, n_samples=1)
    with pytest.raises(ValueError, match="unknown uncertain parameter"):
        uq.monte_carlo({"bogus": (0.0, 1.0)}, grid, n_samples=4)
