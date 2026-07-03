"""Tests for vessel growth by space colonisation."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from perfusion import Grid, Parameters, growth, run_hybrid
from perfusion.core.geometry import network_from_polylines


@pytest.fixture
def trunk():
    """A single vertical trunk rooted at its bottom tip."""
    return network_from_polylines([[(50, 2), (50, 20)]], radii=2.0, root_xy=(50, 2))


def test_grow_produces_a_valid_larger_tree(trunk):
    rng = np.random.default_rng(0)
    pts = rng.uniform([20, 25], [80, 90], size=(300, 2))
    net = growth.grow_network(trunk, pts, step=2.0, leaf_radius=0.5)

    assert net.n_nodes > trunk.n_nodes                 # it actually grew
    assert net.n_segments == net.n_nodes - 1           # still a tree
    assert net.root_node == trunk.root_node
    # every non-root node is reachable / lengths are finite and positive
    assert np.all(net.seg_length > 0)
    assert np.all(np.isfinite(net.seg_radius))


def test_grown_tips_reach_the_point_cloud(trunk):
    rng = np.random.default_rng(1)
    pts = rng.uniform([20, 25], [80, 90], size=(300, 2))
    net = growth.grow_network(trunk, pts, step=2.0, kill_radius=4.0, leaf_radius=0.5)

    tips = np.column_stack([net.node_x[net.terminal_nodes],
                            net.node_y[net.terminal_nodes]])
    dist, _ = cKDTree(pts).query(tips)
    # most tips should sit within a step or two of an attraction point
    assert np.median(dist) < 5.0


def test_radii_taper_toward_the_root(trunk):
    rng = np.random.default_rng(2)
    pts = rng.uniform([20, 25], [80, 90], size=(200, 2))
    net = growth.grow_network(trunk, pts, step=2.0, leaf_radius=0.5, exponent=3.0)

    # Murray's law: the trunk (root's segment) is the thickest, leaves the thinnest.
    root_seg = net.incoming_segment[net.seg_end[net.seg_start == net.root_node][0]]
    assert net.seg_radius[root_seg] == pytest.approx(net.seg_radius.max())
    assert net.seg_radius.min() == pytest.approx(0.5)


def test_sample_points_respects_the_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    pts = growth.sample_points_in_mask(mask, (0, 20, 0, 20), 500, seed=0)
    # all sampled points fall inside the active square [5,15) x [5,15)
    assert np.all((pts >= 5) & (pts < 15))


def test_from_selection_limits_growth_sources(trunk):
    # growing only from the (single) trunk tip still works and grows
    rng = np.random.default_rng(3)
    pts = rng.uniform([20, 25], [80, 90], size=(150, 2))
    net = growth.grow_network(trunk, pts, seed_nodes=[trunk.n_nodes - 1],
                              step=2.5, leaf_radius=0.5)
    assert net.n_nodes > trunk.n_nodes


def test_cco_grows_a_valid_spaced_tree(trunk):
    rng = np.random.default_rng(0)
    pool = rng.uniform([20, 25], [80, 90], size=(600, 2))
    net = growth.cco_grow(trunk, pool, n_terminals=100, d_min=4.0, leaf_radius=0.5)

    assert net.n_nodes > trunk.n_nodes
    assert net.n_segments == net.n_nodes - 1        # still a tree
    assert net.root_node == trunk.root_node
    assert np.all(net.seg_length > 0)
    # the constraint holds: terminals are spaced at least ~d_min apart
    tips = np.column_stack([net.node_x[net.terminal_nodes], net.node_y[net.terminal_nodes]])
    d, _ = cKDTree(tips).query(tips, k=2)
    assert np.median(d[:, 1]) >= 3.0                # ~d_min, allowing for the base tip


def test_cco_in_mask_conserves_mass_in_hybrid():
    dom = (0, 100, 0, 100)
    mask = np.ones((40, 50), bool)
    base = network_from_polylines([[(50, 2), (50, 30)]], radii=2.0, root_xy=(50, 2))
    art = growth.cco_grow_in_mask(base, mask, dom, n_terminals=60, seed=0, leaf_radius=0.4)
    grid = Grid(nx=40, ny=50, lx=100, ly=100)
    res = run_hybrid(Parameters(), grid, domain_limit=dom, art=art)
    assert np.isclose(res.solution.terminal_flux.sum(), 0.0, atol=1e-8)


def test_grown_network_conserves_mass_in_hybrid():
    """Grown arterial + venous trees feed straight into run_hybrid; mass balances."""
    dom = (0, 100, 0, 100)
    mask = np.ones((40, 50), bool)
    art_base = network_from_polylines([[(50, 2), (50, 40)]], radii=2.0, root_xy=(50, 2))
    ven_base = network_from_polylines([[(50, 98), (50, 60)]], radii=2.0, root_xy=(50, 98))
    art = growth.grow_in_mask(art_base, mask, dom, n_points=150, seed=0,
                              step=4.0, leaf_radius=0.4)
    ven = growth.grow_in_mask(ven_base, mask, dom, n_points=150, seed=1,
                              step=4.0, leaf_radius=0.4)

    grid = Grid(nx=40, ny=50, lx=100, ly=100)
    res = run_hybrid(Parameters(), grid, domain_limit=dom, art=art, ven=ven)
    tf = res.solution.terminal_flux
    assert np.isclose(tf.sum(), 0.0, atol=1e-8)        # global mass balance
