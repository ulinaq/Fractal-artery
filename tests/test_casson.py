"""Casson (yield-stress) blood: the exact tube law, its secant viscosity and the
accelerated fixed-point network solve."""

from types import SimpleNamespace

import numpy as np
import pytest

from perfusion import Grid, Parameters
from perfusion.core import pipeline
from perfusion.core.viscosity import (
    CASSON_MAX_FACTOR,
    casson_factor,
    casson_flow_ratio,
    solve_shear_thinning,
)
from perfusion.io import results_summary


def test_casson_tube_flow_ratio():
    assert casson_flow_ratio(0.0) == pytest.approx(1.0)               # no yield: Newtonian
    assert casson_flow_ratio(1.0) == pytest.approx(0.0, abs=1e-12)    # yield at the wall
    assert casson_flow_ratio(3.0) == 0.0                              # plug: no flow
    xi = np.linspace(0, 1, 101)
    assert np.all(np.diff(casson_flow_ratio(xi)) < 0)


def test_casson_factor_is_the_exact_tube_law():
    """At wall stress τ_w the factor taken from the flow it carries is 1 / ratio(τ_y/τ_w):
    Poiseuille with it gives exactly the Casson tube flow."""
    tau_y, mu = 0.3, 2.0
    tw = np.array([0.31, 0.4, 0.6, 1.0, 3.0, 30.0, 3e4])
    ratio = casson_flow_ratio(tau_y / tw)
    shear = tw * ratio / mu                     # Newtonian-equivalent wall rate of Q
    f = casson_factor(shear, tau_y / mu)
    assert np.allclose(f, np.minimum(1.0 / ratio, CASSON_MAX_FACTOR), rtol=1e-9)
    assert f[0] == CASSON_MAX_FACTOR                               # just above yield: capped
    assert f[-1] == pytest.approx(1.0, abs=0.03)                   # high stress: μ_c
    assert np.all(np.diff(f) < 0)
    assert casson_factor(0.0, 0.1) == CASSON_MAX_FACTOR             # no flow: plug
    assert np.allclose(casson_factor([0.0, 1e-3, 5.0], 0.0), 1.0)   # τ_y = 0: Newtonian
    per_seg = casson_factor([1.0, 1.0], [0.0, 0.1])                 # per-segment yield
    assert per_seg[0] == 1.0 < per_seg[1]
    with pytest.raises(ValueError, match="Casson"):
        casson_factor(1.0, -0.1)


def _series(p_total, r_other, length=10.0, radius=1.0):
    """One vessel (Poiseuille R0 = 8L/πr⁴ per unit μ) in series with a Newtonian
    resistor, driven by ``p_total``: the smallest network with a Casson segment."""
    net = SimpleNamespace(n_segments=1, seg_radius=np.array([radius]))
    empty = SimpleNamespace(n_segments=0, seg_radius=np.zeros(0))
    r0 = 8.0 * length / (np.pi * radius**4)

    def solve(a, v, visc):
        q = p_total / (r0 * visc[0][0] + r_other)
        return SimpleNamespace(q_a=np.array([q]), q_v=np.zeros(0))
    return net, empty, solve, r0


@pytest.mark.parametrize("r_other", [0.0, 5.0, 500.0])
def test_single_vessel_carries_the_exact_casson_flow(r_other):
    length, radius, tau_y = 10.0, 1.0, 0.02
    art, ven, solve, r0 = _series(2.0, r_other, length, radius)
    sol, (mu_a, _), info = solve_shear_thinning(solve, art, ven, 1.0, law="casson",
                                                yield_stress=tau_y)
    assert info["converged"]
    assert info["plugged"] == 0
    q = sol.q_a[0]
    dp = r0 * mu_a[0] * q                                      # pressure drop on the vessel
    tw = dp * radius / (2 * length)
    exact = np.pi * radius**3 * tw / 4 * casson_flow_ratio(tau_y / tw)
    assert q == pytest.approx(exact, rel=1e-5)
    assert q < 2.0 / (r0 + r_other)                            # slower than Newtonian


def test_below_yield_the_vessel_plugs():
    art, ven, solve, _ = _series(0.5, 0.0)                     # τ_w = 0.025 < τ_y
    sol, _, info = solve_shear_thinning(solve, art, ven, 1.0, law="casson",
                                        yield_stress=0.05)
    assert info["plugged"] == 1
    assert info["factor_max"] == pytest.approx(CASSON_MAX_FACTOR)
    newtonian = solve(art, ven, (np.ones(1), np.zeros(0))).q_a[0]
    assert sol.q_a[0] < 1e-3 * newtonian


def test_bad_law_and_yield_are_rejected():
    art, ven, solve, _ = _series(1.0, 0.0)
    with pytest.raises(ValueError, match="unknown blood law"):
        solve_shear_thinning(solve, art, ven, 1.0, law="bingham")
    with pytest.raises(ValueError, match="yield_stress"):
        solve_shear_thinning(solve, art, ven, 1.0, law="casson", yield_stress=-1.0)


_GRID = Grid(nx=40, ny=46, lx=140, ly=160)


def _run(**kw):
    return pipeline.run_hybrid(Parameters(r_min_frac=0.3, **kw), _GRID)


def _casson(**kw):
    return _run(shear_thinning=True, blood_law="casson", **kw)


def test_zero_yield_is_newtonian():
    base, flat = _run(), _casson(casson_yield=0.0)
    assert np.allclose(flat.solution.q_a, base.solution.q_a, rtol=1e-12, atol=1e-14)
    assert flat.rheology["plugged"] == 0
    assert flat.rheology["factor_max"] == pytest.approx(1.0)


def test_casson_network_is_self_consistent_and_conserves_mass():
    res = _casson()
    sol, info = res.solution, res.rheology
    assert info["converged"]
    assert info["law"] == "casson"
    n_art = res.art.terminal_nodes.size
    assert np.isclose(sol.terminal_flux[:n_art].sum(), -sol.terminal_flux[n_art:].sum(),
                      rtol=1e-9)
    # every arterial segment carries the exact Casson flow for its own pressure drop
    art = res.art
    dp = np.abs(sol.p_a[art.seg_start] - sol.p_a[art.seg_end])
    tw = dp * art.seg_radius / (2 * art.seg_length)
    exact = np.pi * art.seg_radius**3 * tw / 4 * casson_flow_ratio(0.001 / tw)
    assert np.allclose(np.abs(sol.q_a), exact, rtol=1e-4, atol=1e-12)
    s = results_summary(res)
    assert s["plugged_vessels"] == info["plugged"]
    assert "plugged_vessels" not in results_summary(_run(shear_thinning=True))  # Carreau


def test_casson_flow_stops_below_a_yield_pressure():
    """Q_casson / Q_newtonian falls towards 0 as the pressure drops, with vessels
    plugging; above yield √Q is linear in √Δp (the Casson signature)."""
    def inflow(res):
        return results_summary(res)["arterial_inflow"]
    ps = (0.25, 1.0, 5.0, 20.0)
    ratio = [inflow(_casson(p_in=p)) / inflow(_run(p_in=p)) for p in ps]
    assert ratio == sorted(ratio)
    assert ratio[0] < 1e-2 < 0.5 < ratio[-1] < 1.0
    assert _casson(p_in=0.25).rheology["plugged"] > 0
    hi = [2.0, 5.0, 10.0, 20.0]
    q = [inflow(_casson(p_in=p)) for p in hi]
    fit = np.polyfit(np.sqrt(hi), np.sqrt(q), 1, full=True)
    assert fit[1][0] < 1e-3 * np.sum(np.sqrt(q)) ** 2           # √Q linear in √p
    assert fit[0][1] < 0                                         # ... with a yield offset

