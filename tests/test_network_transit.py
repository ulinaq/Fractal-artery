"""Network-transit kinetics for the direct-connection model (core.network_transit)."""

import numpy as np

from perfusion.core import network_transit as nt
from perfusion.core.direct import connect_terminals, solve_direct


def _solve(art, ven, small_params):
    conns = connect_terminals(art, ven, rho_c=0.0, viscosity=small_params.mu)
    sol = solve_direct(art, ven, conns, small_params.p_in, small_params.p_out,
                       viscosity=small_params.mu)
    return conns, sol


def test_arrival_times_increase_along_flow(art, ven, small_params):
    conns, sol = _solve(art, ven, small_params)
    t_art, t_ven = nt.arrival_times(art, ven, conns, sol)

    # the arterial inlet is the bolus source at t=0; terminals are strictly later
    for r in art.inlet_nodes:
        assert t_art[int(r)] == 0.0
    reached = t_art[art.terminal_nodes]
    assert np.all(reached[np.isfinite(reached)] > 0.0)
    # veins are downstream of arteries, so the mean venous arrival is later
    fa = t_art[np.isfinite(t_art)]
    fv = t_ven[np.isfinite(t_ven)]
    assert fv.mean() > fa.mean()


def test_transit_animation_sweeps_in_order(art, ven, small_params):
    conns, sol = _solve(art, ven, small_params)
    anim = nt.transit_animation(art, ven, conns, sol, n_frames=40)

    n_elem = art.n_segments + ven.n_segments + len(conns)
    assert anim["lines"].shape == (n_elem, 4)
    assert anim["frames"].shape == (40, n_elem)
    assert set(np.unique(anim["kind"])) <= {0, 1, 2}

    # normalised bolus: reachable elements each peak near 1
    reach = np.isfinite(anim["arrivals"])
    peaks = anim["frames"][:, reach].max(axis=0)
    assert peaks.min() > 0.9
    # the frame at which each element peaks tracks its arrival time (bolus sweep)
    peak_frame = anim["frames"][:, reach].argmax(axis=0)
    assert np.corrcoef(anim["arrivals"][reach], peak_frame)[0, 1] > 0.9
    # unreached elements never light up
    assert np.all(anim["frames"][:, ~reach] == 0.0)


def test_zero_flow_elements_stay_dark(art, ven, small_params):
    # a bolus can only reach elements that carry flow; the finite-arrival set must
    # be exactly the elements the animation ever lights.
    conns, sol = _solve(art, ven, small_params)
    anim = nt.transit_animation(art, ven, conns, sol, n_frames=30)
    ever_lit = anim["frames"].max(axis=0) > 0
    assert np.array_equal(ever_lit, np.isfinite(anim["arrivals"]))
