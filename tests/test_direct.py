"""Tests for the direct-connection microcirculation model (Paper C)."""

import pytest

from perfusion import direct


def test_relative_capillary_resistance():
    # n = 1 generation -> gamma = 2^(2/3)
    assert direct.relative_capillary_resistance(1) == pytest.approx(2 ** (2 / 3))
    # more generations increase the effective resistance
    assert direct.relative_capillary_resistance(3) > direct.relative_capillary_resistance(1)


def test_connect_terminals_within_radius(art, ven):
    conns = direct.connect_terminals(art, ven, rho_c=60.0)
    assert len(conns) > 0
    for c in conns:
        assert c.arterial_terminal in art.terminal_nodes
        assert c.venous_terminal in ven.terminal_nodes
        assert c.resistance > 0
        assert c.length >= 0


def test_connect_terminals_nearest_fallback(art, ven):
    # rho_c = 0 forces the single-nearest-neighbour fallback for every terminal
    conns = direct.connect_terminals(art, ven, rho_c=0.0)
    assert len(conns) == art.terminal_nodes.size


def test_connect_terminals_matches_bruteforce(art, ven):
    # The KD-tree radius search must agree with an O(T^2) brute-force count.
    import numpy as np

    rho_c = 50.0
    conns = direct.connect_terminals(art, ven, rho_c=rho_c)
    ta, tv = art.terminal_nodes, ven.terminal_nodes
    ax, ay = art.node_x[ta], art.node_y[ta]
    vx, vy = ven.node_x[tv], ven.node_y[tv]
    brute = 0
    for i in range(len(ta)):
        near = np.count_nonzero(np.hypot(vx - ax[i], vy - ay[i]) <= rho_c)
        brute += max(near, 1)                 # nearest-neighbour fallback when none
    assert len(conns) == brute


def test_solve_direct_conserves_mass(art, ven):
    conns = direct.connect_terminals(art, ven, rho_c=60.0)
    sol = direct.solve_direct(art, ven, conns, p_in=75.0, p_out=20.0)
    # arterial trunk inflow == total connection flow == venous trunk outflow
    assert sol.q_a[0] == pytest.approx(sol.q_c.sum(), rel=1e-6)
    assert sol.q_a[0] == pytest.approx(sol.q_v[0], rel=1e-6)
    assert sol.q_a[0] > 0


def test_solve_direct_pressure_bounds(art, ven):
    conns = direct.connect_terminals(art, ven, rho_c=60.0)
    sol = direct.solve_direct(art, ven, conns, p_in=75.0, p_out=20.0)
    assert sol.p_a[art.root_node] == pytest.approx(75.0)
    assert sol.p_v[ven.root_node] == pytest.approx(20.0)
    # all pressures lie within the driving range
    assert sol.p_a.max() <= 75.0 + 1e-6
    assert sol.p_v.min() >= 20.0 - 1e-6
