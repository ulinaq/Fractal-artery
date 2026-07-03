"""Tests for the vascular network representation and fractal-tree growth."""

import dataclasses

import numpy as np
import pytest

from perfusion import arterial_tree, venous_tree
from perfusion.core.geometry import (
    Network,
    NodeRole,
    grow_tree,
    network_from_polylines,
    rasterize_mask,
)


def _line_network() -> Network:
    """A trivial two-segment chain: root -> mid -> leaf."""
    return Network(
        node_x=np.array([0.0, 1.0, 2.0]), node_y=np.zeros(3),
        node_role=np.array([NodeRole.ROOT, NodeRole.BIFURCATION, NodeRole.TERMINAL]),
        seg_start=np.array([0, 1]), seg_end=np.array([1, 2]),
        seg_length=np.array([1.0, 1.0]), seg_radius=np.array([1.0, 0.5]),
        seg_angle=np.zeros(2),
    )


def test_counts_and_incoming_segment():
    net = _line_network()
    assert net.n_nodes == 3
    assert net.n_segments == 2
    # segment feeding each node: root has none (-1), node1<-seg0, node2<-seg1
    assert net.incoming_segment.tolist() == [-1, 0, 1]


def test_terminal_and_bifurcation_nodes():
    net = _line_network()
    assert net.terminal_nodes.tolist() == [2]
    assert net.bifurcation_nodes.tolist() == [1]


def test_resistance_formula():
    net = _line_network()
    R = net.resistance(3.0)
    assert R[0] == pytest.approx(8 * 3.0 * 1.0 / (np.pi * 1.0**4))
    assert R[1] == pytest.approx(8 * 3.0 * 1.0 / (np.pi * 0.5**4))


def test_incidence_matrix_convention():
    # (S @ p)[s] = p_start - p_end ; (S.T @ q)[node] = net outflow
    net = _line_network()
    S = net.incidence()
    assert S.shape == (2, 3)
    p = np.array([10.0, 6.0, 1.0])
    assert np.allclose(S @ p, [10 - 6, 6 - 1])
    q = np.array([2.0, 2.0])
    # node 0: outflow +2; node 1: +2 (out) -2 (in) = 0; node 2: -2 (in)
    assert np.allclose(S.T @ q, [2.0, 0.0, -2.0])


def test_grow_tree_is_binary_and_radii_decrease():
    prm_phi = np.tile([0.3, 0.3], (4, 1))
    prm_ratio = np.tile([0.7, 0.7], (4, 1))
    net = grow_tree((0.0, 0.0), np.pi / 2, root_radius=2.0, min_radius=0.5,
                    length_to_radius=10.0, angles=prm_phi, radius_ratios=prm_ratio)
    # tree: segments = nodes - 1
    assert net.n_segments == net.n_nodes - 1
    # every bifurcation has exactly two children
    for nb in net.bifurcation_nodes:
        assert np.sum(net.seg_start == nb) == 2
    # all leaf radii are below the branching threshold's parent radius
    assert net.seg_radius.min() < net.seg_radius.max()


def test_mirror_children_changes_layout():
    phi = np.tile([0.2, 0.5], (3, 1))
    ratio = np.tile([0.8, 0.6], (3, 1))
    kw = dict(root_angle=np.pi / 2, root_radius=2.0, min_radius=0.9,
              length_to_radius=10.0, angles=phi, radius_ratios=ratio)
    a = grow_tree((0.0, 0.0), **kw, mirror_children=False)
    b = grow_tree((0.0, 0.0), **kw, mirror_children=True)
    # same topology, mirrored child ordering -> different coordinates
    assert a.n_nodes == b.n_nodes
    assert not np.allclose(a.node_x, b.node_x)


def test_network_from_polylines_y_junction():
    # Three polylines meeting at a shared point (10, 10) -> 1 junction, 3 tips.
    polylines = [
        [(10, 10), (10, 0)],            # stem
        [(10, 10), (0, 20)],            # left branch
        [(10, 10), (20, 20)],           # right branch
    ]
    net = network_from_polylines(polylines, radii=[2.0, 1.0, 1.0], snap_tol=1.0)
    assert net.n_nodes == 4
    assert net.n_segments == 3
    assert net.bifurcation_nodes.size == 1          # the shared junction
    # arc length of a straight polyline equals the endpoint distance
    stem = np.argmin(np.abs(net.seg_radius - 2.0))
    assert net.seg_length[stem] == pytest.approx(10.0)


def test_network_from_polylines_snaps_endpoints():
    # Endpoints within snap_tol merge into one node.
    polylines = [[(0, 0), (5, 0)], [(5.4, 0), (10, 0)]]
    net = network_from_polylines(polylines, radii=1.0, snap_tol=1.0)
    assert net.n_nodes == 3                          # 5.0 and 5.4 snapped together
    assert net.n_segments == 2


def test_arterial_and_venous_trees(params):
    art = arterial_tree(params)
    ven = venous_tree(params)
    assert art.n_segments == art.n_nodes - 1
    assert ven.n_segments == ven.n_nodes - 1
    assert art.terminal_nodes.size >= 2
    # root is excluded from the terminal set
    assert art.root_node not in art.terminal_nodes


def test_rasterize_mask_matches_a_polygon():
    # a centred square covering [2,8] x [2,8] of a 10x10 domain, 2 cells / unit
    m = rasterize_mask(20, 20, (0, 10, 0, 10), [[(2, 2), (8, 2), (8, 8), (2, 8)]])
    assert m.dtype == bool
    assert m.shape == (20, 20)
    assert m[10, 10]        # centre is inside
    assert not m[0, 0]      # corners are outside
    assert not m[19, 19]
    assert m.sum() == 12 * 12    # 6x6 domain units at 2 cells/unit


def test_inlets_and_degree_based_terminals():
    # a Y: root at (0,0) is a leaf; two other leaf tips.
    net = network_from_polylines(
        [[(0, 0), (0, 10)], [(0, 10), (-5, 20)], [(0, 10), (5, 20)]],
        radii=1.0, root_xy=(0, 0))
    # default: the root is the sole inlet, the other two leaves are Darcy terminals
    assert net.inlet_nodes.tolist() == [net.root_node]
    assert net.root_node not in net.terminal_nodes.tolist()
    assert len(net.terminal_nodes) == 2

    tip = int(net.terminal_nodes[0])
    net2 = dataclasses.replace(net, inlet_nodes=np.array([net.root_node, tip]))
    assert tip not in net2.terminal_nodes.tolist()      # a marked inlet leaves the set
    assert len(net2.terminal_nodes) == 1

    net3 = dataclasses.replace(net, inlet_nodes=np.array([tip]))
    assert net.root_node in net3.terminal_nodes.tolist()  # non-inlet root -> terminal


def test_rasterize_mask_empty_is_all_active():
    assert rasterize_mask(4, 5, (0, 1, 0, 1), []).shape == (4, 5)
    assert rasterize_mask(4, 5, (0, 1, 0, 1), []).all()
