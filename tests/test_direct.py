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


def _trees_3d():
    from perfusion.core.geometry import grow_tree_3d

    art = grow_tree_3d((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), root_radius=1.0,
                       min_radius=0.3, max_generations=4)
    ven = grow_tree_3d((6.0, 0.0, 0.0), (0.0, 0.0, 1.0), root_radius=1.2,
                       min_radius=0.3, max_generations=4, twist=1.0)
    return art, ven


def test_connect_terminals_uses_3d_distance():
    """3-D trees link by Euclidean distance in x, y *and* z (a planar distance would
    pair terminals that are far apart in depth)."""
    import numpy as np

    art, ven = _trees_3d()
    conns = direct.connect_terminals(art, ven, rho_c=0.0)
    vxyz = ven.positions(ven.terminal_nodes)
    for c in conns:
        a = art.positions([c.arterial_terminal])[0]
        d = np.linalg.norm(vxyz - a, axis=1)
        assert c.length == pytest.approx(d.min())          # the nearest in 3-D
        assert c.length == pytest.approx(
            np.linalg.norm(a - ven.positions([c.venous_terminal])[0]))


def test_solve_direct_conserves_mass_in_3d():
    art, ven = _trees_3d()
    conns = direct.connect_terminals(art, ven, rho_c=8.0)
    sol = direct.solve_direct(art, ven, conns, p_in=10.0, p_out=0.0)
    assert sol.q_a[0] > 0
    assert sol.q_a[0] == pytest.approx(sol.q_c.sum(), rel=1e-6)
    assert sol.q_a[0] == pytest.approx(sol.q_v[0], rel=1e-6)


def test_connect_terminals_rejects_mixed_dimensions(art):
    _, ven3 = _trees_3d()
    with pytest.raises(ValueError, match="both be 2-D or both 3-D"):
        direct.connect_terminals(art, ven3, rho_c=1.0)


def test_network_transit_animation_in_3d():
    from perfusion.core import network_transit as nt

    art, ven = _trees_3d()
    conns = direct.connect_terminals(art, ven, rho_c=0.0)
    sol = direct.solve_direct(art, ven, conns, p_in=10.0, p_out=0.0)
    anim = nt.transit_animation(art, ven, conns, sol, n_frames=6)
    assert anim["lines"].shape == (art.n_segments + ven.n_segments + len(conns), 6)
    assert anim["frames"].shape == (6, anim["lines"].shape[0])


def test_solve_direct_honours_every_inlet_and_outlet(art, ven):
    # The direct model pinned only root_node: a second inlet gave a non-square system.
    import dataclasses

    import numpy as np

    ta, tv = art.terminal_nodes, ven.terminal_nodes
    art2 = dataclasses.replace(art, inlet_nodes=np.array([art.root_node, int(ta[0])]))
    ven2 = dataclasses.replace(ven, inlet_nodes=np.array([ven.root_node, int(tv[0])]))
    conns = direct.connect_terminals(art2, ven2, rho_c=60.0)
    sol = direct.solve_direct(art2, ven2, conns, p_in=75.0, p_out=20.0)
    assert np.allclose(sol.p_a[art2.inlet_nodes], 75.0)
    assert np.allclose(sol.p_v[ven2.inlet_nodes], 20.0)
    assert sol.q_c.sum() > 0


def test_solve_direct_with_a_leaf_inlet_and_reversed_segment(art, ven):
    import dataclasses

    import numpy as np

    conns = direct.connect_terminals(art, ven, rho_c=60.0)
    ref = direct.solve_direct(art, ven, conns, p_in=75.0, p_out=20.0)
    # a reversed terminal segment must not change the physics
    s = int(art.leaf_segments(art.terminal_nodes[:1])[0][0])
    ss, se = art.seg_start.copy(), art.seg_end.copy()
    ss[s], se[s] = se[s], ss[s]
    art_r = dataclasses.replace(art, seg_start=ss, seg_end=se)
    sol = direct.solve_direct(art_r, ven, conns, p_in=75.0, p_out=20.0)
    assert np.allclose(sol.q_c, ref.q_c, rtol=1e-9)
    # an inlet at a leaf (the BFS root then becomes a terminal) solves and conserves
    leaf = int(art.terminal_nodes[0])
    art_l = dataclasses.replace(art, inlet_nodes=np.array([leaf]))
    conns_l = direct.connect_terminals(art_l, ven, rho_c=60.0)
    sol_l = direct.solve_direct(art_l, ven, conns_l, p_in=75.0, p_out=20.0)
    assert sol_l.q_c.sum() == pytest.approx(sol_l.q_v[0], rel=1e-6)
    assert sol_l.p_a[leaf] == pytest.approx(75.0)
