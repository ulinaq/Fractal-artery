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


def test_density_weighted_sampling_concentrates_points():
    nx, ny = 40, 40
    mask = np.ones((nx, ny), bool)
    dom = (0.0, 40.0, 0.0, 40.0)
    density = np.zeros((nx, ny))
    density[nx // 2:, ny // 2:] = 1.0                 # demand only in the upper-right quadrant

    uniform = growth.sample_points_in_mask(mask, dom, 2000, seed=1)
    weighted = growth.sample_points_in_mask(mask, dom, 2000, seed=1, density=density)

    def frac(p):
        return np.mean((p[:, 0] > 20) & (p[:, 1] > 20))

    assert frac(uniform) < 0.4                        # ~1/4 of the domain
    assert frac(weighted) > 0.95                      # concentrated in the hotspot
    # density=None is identical to the uniform sampler (regression preserved)
    same = growth.sample_points_in_mask(mask, dom, 300, seed=7)
    none = growth.sample_points_in_mask(mask, dom, 300, seed=7, density=None)
    assert np.allclose(same, none)


def test_density_growth_biases_terminals(trunk):
    nx, ny = 50, 50
    mask = np.ones((nx, ny), bool)
    dom = (0.0, 50.0, 0.0, 50.0)
    density = np.zeros((nx, ny))
    density[nx // 2:, :] = 1.0                         # demand in the right half
    net = growth.grow_in_mask(trunk, mask, dom, n_points=400, seed=2, density=density, step=2.0)
    tn = net.terminal_nodes
    assert np.mean(net.node_x[tn] > 25) > 0.6          # terminals skew to the high-demand half


def test_flat_density_falls_back_and_validates():
    mask = np.ones((10, 10), bool)
    dom = (0.0, 10.0, 0.0, 10.0)
    pts = growth.sample_points_in_mask(mask, dom, 50, seed=0, density=np.zeros((10, 10)))
    assert pts.shape == (50, 2)                        # empty map → uniform fallback, no crash
    with pytest.raises(ValueError, match="must match mask"):
        growth.sample_points_in_mask(mask, dom, 10, density=np.ones((3, 3)))


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


def test_obstacles_keep_growth_out(trunk):
    nx, ny = 50, 50
    mask = np.ones((nx, ny), bool)
    dom = (0.0, 50.0, 0.0, 50.0)
    obstacle = [[(10, 10), (40, 10), (40, 40), (10, 40)]]   # central no-grow box
    # sampled attraction points avoid the obstacle
    pts = growth.sample_points_in_mask(
        growth.mask_without_obstacles(mask, dom, obstacle), dom, 500, seed=1)
    inside = np.mean((pts[:, 0] > 10) & (pts[:, 0] < 40) & (pts[:, 1] > 10) & (pts[:, 1] < 40))
    assert inside == 0.0
    # grown terminals avoid the obstacle too
    net = growth.grow_in_mask(trunk, mask, dom, n_points=400, seed=2,
                              obstacles=obstacle, step=2.0)
    tn = net.terminal_nodes
    tx, ty = net.node_x[tn], net.node_y[tn]
    assert np.mean((tx > 12) & (tx < 38) & (ty > 12) & (ty < 38)) < 0.05


def _trunk_3d():
    from perfusion.core.geometry import grow_tree_3d
    return grow_tree_3d((30, 30, 3), (0, 0, 1), root_radius=1.5, min_radius=0.0,
                        length_to_radius=6, max_generations=2)


def test_sample_points_in_3d_mask_stay_inside():
    from perfusion.growth import sample_points_in_mask
    from perfusion.treegen import DOMAIN_3D, shape_mask_3d
    mask = shape_mask_3d("ellipsoid", 20, 20, 20)
    pts = sample_points_in_mask(mask, DOMAIN_3D, 300, seed=1)
    assert pts.shape == (300, 3)
    c = np.array([30.0, 30.0, 30.0])
    r = 0.85 * 30 * 1.1                        # semi-axis + one cell of slack
    assert np.all(np.linalg.norm(pts - c, axis=1) <= r)


def test_space_colonisation_grows_a_3d_tree_into_the_tissue():
    from perfusion.growth import grow_network, sample_points_in_mask
    from perfusion.treegen import DOMAIN_3D, shape_mask_3d
    base = _trunk_3d()
    pts = sample_points_in_mask(shape_mask_3d("box", 16, 16, 16), DOMAIN_3D, 200, seed=0)
    net = grow_network(base, pts, max_iter=30)
    assert net.is_3d
    assert net.n_segments > base.n_segments
    assert np.ptp(net.node_z) > np.ptp(base.node_z)    # it filled the volume
    # Murray-tapered: no child thicker than its parent
    parent_seg = net.incoming_segment[net.seg_start]
    has_parent = parent_seg >= 0
    assert np.all(net.seg_radius[has_parent] <= net.seg_radius[parent_seg[has_parent]] + 1e-12)


def test_cco_grows_in_3d_and_rejects_wrong_dimension():
    from perfusion.growth import cco_grow
    base = _trunk_3d()
    rng = np.random.default_rng(0)
    pool = rng.uniform(5, 55, size=(300, 3))
    net = cco_grow(base, pool, n_terminals=25, seed=0)
    assert net.is_3d
    assert net.n_segments >= base.n_segments + 25          # one new edge per terminal
    assert net.terminal_nodes.size > base.terminal_nodes.size
    with pytest.raises(ValueError, match=r"\(M, 3\)"):
        cco_grow(base, pool[:, :2], n_terminals=5)
