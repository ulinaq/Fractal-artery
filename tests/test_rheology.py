"""Shear-thinning (Carreau) blood: the viscosity law and the fixed-point network solve."""

import numpy as np
import pytest

from perfusion import Grid, Parameters
from perfusion.core import pipeline
from perfusion.core.viscosity import (
    CARREAU_MU0_RATIO,
    carreau_factor,
    solve_shear_thinning,
    wall_shear_rate,
)
from perfusion.io import results_summary


def test_carreau_limits_and_monotone():
    g = np.logspace(-6, 6, 50)
    f = carreau_factor(g, lam=1.0)
    assert f[0] == pytest.approx(CARREAU_MU0_RATIO, rel=1e-6)    # at rest: mu_0
    assert f[-1] == pytest.approx(1.0, abs=1e-2)                 # high shear: mu_inf
    assert np.all(np.diff(f) < 0)                                # shear-thinning
    # power-law range: mu ∝ g^(n-1)
    k = np.polyfit(np.log(g[30:40]), np.log(f[30:40] - 1.0), 1)[0]
    assert k == pytest.approx(0.3568 - 1.0, abs=0.02)
    assert np.allclose(carreau_factor(g, lam=0.0), CARREAU_MU0_RATIO)
    assert np.allclose(carreau_factor(g, mu0_ratio=1.0), 1.0)    # Newtonian
    with pytest.raises(ValueError, match="Carreau"):
        carreau_factor(g, n=1.5)


def test_wall_shear_rate_poiseuille():
    assert wall_shear_rate(np.pi, 1.0) == pytest.approx(4.0)
    assert wall_shear_rate(-np.pi, 2.0) == pytest.approx(0.5)


_GRID = Grid(nx=40, ny=46, lx=140, ly=160)


def _run(**kw):
    return pipeline.run_hybrid(Parameters(r_min_frac=0.3, **kw), _GRID)


def test_newtonian_limit_is_identical():
    base = _run()
    flat = _run(shear_thinning=True, carreau_mu0_ratio=1.0)
    assert np.allclose(flat.solution.q_a, base.solution.q_a, rtol=1e-12, atol=1e-14)
    assert flat.rheology["factor_max"] == pytest.approx(1.0)


def test_shear_thinning_self_consistent_and_conserves_mass():
    res = _run(shear_thinning=True)
    sol, info = res.solution, res.rheology
    assert info["converged"]
    n_art = res.art.terminal_nodes.size
    assert np.isclose(sol.terminal_flux[:n_art].sum(), -sol.terminal_flux[n_art:].sum(),
                      rtol=1e-9)
    # the solved flows reproduce the viscosities they were solved with (Poiseuille check)
    f = carreau_factor(wall_shear_rate(sol.q_a, res.art.seg_radius), lam=3000.0)
    dp = sol.p_a[res.art.seg_start] - sol.p_a[res.art.seg_end]
    r0 = res.art.resistance(1.0)
    assert np.allclose(dp, r0 * f * sol.q_a, rtol=1e-4, atol=1e-10)
    assert 1.0 < info["factor_min"] < info["factor_max"] <= CARREAU_MU0_RATIO
    s = results_summary(res)
    assert s["viscosity_factor_max"] == pytest.approx(info["factor_max"])
    assert "viscosity_factor_max" not in results_summary(_run())


def test_shear_thinning_flow_grows_faster_than_pressure():
    q = [results_summary(_run(p_in=p))["arterial_inflow"] for p in (5.0, 10.0)]
    qs = [results_summary(_run(p_in=p, shear_thinning=True))["arterial_inflow"]
          for p in (5.0, 10.0)]
    assert q[1] / q[0] == pytest.approx(2.0, rel=1e-9)            # Newtonian: linear
    assert qs[1] / qs[0] > 2.1                                    # thinning: superlinear
    assert qs[1] < q[1]                                           # thicker than mu_inf


def test_composes_with_fahraeus_lindqvist():
    res = _run(shear_thinning=True, fahraeus_lindqvist=True, radius_to_um=20.0)
    assert res.rheology["converged"]
    fl = _run(fahraeus_lindqvist=True, radius_to_um=20.0)
    assert results_summary(res)["arterial_inflow"] < results_summary(fl)["arterial_inflow"]


def test_nonconvergence_warns():
    art, ven, (resp, _, _) = pipeline.darcy_response_for(Parameters(r_min_frac=0.3), _GRID)

    def solve(a, v, visc):
        return pipeline.network.solve_linear(a, v, resp, 10.0, 0.0, visc)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        _, _, info = solve_shear_thinning(solve, art, ven, 1.0, lam=3000.0, max_iter=2)
    assert not info["converged"]
    assert info["iterations"] == 2
