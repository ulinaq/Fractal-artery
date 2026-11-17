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


# ── Darcy-response reuse across sweep samples ──────────────────────────────────
def _count_factorisations(monkeypatch):
    from perfusion.core import darcy
    calls = {"n": 0}
    real = darcy.terminal_response

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(darcy, "terminal_response", counting)
    return calls


def test_monte_carlo_reuses_the_darcy_factorisation_when_geometry_is_fixed(monkeypatch):
    grid = Grid(nx=30, ny=34, lx=140, ly=160)
    ranges = {"p_in": (5.0, 15.0), "mu": (0.8, 1.4)}
    calls = _count_factorisations(monkeypatch)
    fast = uq.monte_carlo(ranges, grid, n_samples=8, base={"r_min_frac": 0.5})
    assert calls["n"] == 1                      # one LU for all 8 samples
    assert fast.mean["arterial_inflow"] > 0
    calls["n"] = 0
    uq.monte_carlo({**ranges, "permeability": (0.9, 1.1)}, grid, n_samples=8,
                   base={"r_min_frac": 0.5})
    assert calls["n"] == 8                      # K varies -> must refactorise each time


def test_reused_response_gives_identical_results():
    """Reuse must be numerically identical to recomputing per sample."""
    from perfusion import Parameters, run_hybrid
    from perfusion.core.pipeline import darcy_response_for
    grid = Grid(nx=30, ny=34, lx=140, ly=160)
    a, v, resp = darcy_response_for(Parameters(r_min_frac=0.5), grid)
    for p_in, mu in [(5.0, 1.0), (12.0, 1.3)]:
        prm = Parameters(p_in=p_in, mu=mu, r_min_frac=0.5)
        fresh = run_hybrid(prm, grid)
        reused = run_hybrid(prm, grid, art=a, ven=v, darcy_response=resp)
        assert np.allclose(fresh.pressure, reused.pressure, rtol=1e-12, atol=1e-14)
        assert np.allclose(fresh.solution.terminal_flux, reused.solution.terminal_flux,
                           rtol=1e-12, atol=1e-14)


def test_tree_depth_sweeps_still_rebuild_the_geometry(monkeypatch):
    grid = Grid(nx=24, ny=28, lx=140, ly=160)
    calls = _count_factorisations(monkeypatch)
    res = uq.monte_carlo({"r_min_frac": (0.4, 0.7)}, grid, n_samples=4)
    assert calls["n"] == 4                      # terminals move -> no reuse
    assert res.std["arterial_inflow"] > 0
    assert uq.geometry_fixed(["p_in", "mu", "hematocrit"])
    assert not uq.geometry_fixed(["permeability"])
    assert not uq.geometry_fixed(["r_min_frac"])
    assert uq.geometry_fixed(["r_min_frac"], art=object(), ven=object())   # fixed trees


def test_calibration_reuses_the_factorisation_and_recovers_p_in(monkeypatch):
    from perfusion import Parameters, inverse, run_hybrid
    grid = Grid(nx=24, ny=28, lx=140, ly=160)
    true = run_hybrid(Parameters(p_in=12.0, r_min_frac=0.5), grid)
    target = float(true.solution.terminal_flux.clip(min=0).sum())
    calls = _count_factorisations(monkeypatch)
    res = inverse.calibrate(target, "p_in", grid, (1.0, 30.0), base={"r_min_frac": 0.5})
    assert calls["n"] == 1                      # every evaluation reused one LU
    assert res.value == pytest.approx(12.0, rel=1e-3)
