"""Horton–Strahler morphometry (core.morphometry)."""

from itertools import pairwise

import numpy as np
import pytest

from perfusion import Parameters, arterial_tree
from perfusion.core import morphometry as m
from perfusion.core.geometry import network_from_edges


def _complete_binary_tree(depth: int):
    """Build a rooted complete binary tree of the given depth (root = node 0)."""
    coords = [(0.0, 0.0)]
    edges = []
    level, nid = [0], 1
    for d in range(depth):
        nxt = []
        for parent in level:
            for k in (-1, 1):
                coords.append((coords[parent][0] + k * 2.0 ** (depth - d),
                               coords[parent][1] - 1.0))
                edges.append((parent, nid, 1.0, 0.8 ** d))   # radius tapers with depth
                nxt.append(nid)
                nid += 1
        level = nxt
    nx = [c[0] for c in coords]
    ny = [c[1] for c in coords]
    return network_from_edges(nx, ny, edges, root=0)


def test_strahler_orders_on_complete_binary_tree():
    net = _complete_binary_tree(3)             # 8 leaves, 4 mid, 2 near-root segments
    s = m.strahler_summary(net)
    assert s["max_order"] == 3
    assert s["counts"] == {1: 8, 2: 4, 3: 2}
    assert s["bifurcation_ratio"] == pytest.approx(2.0, abs=1e-6)   # Horton R_B = 2 exactly


def test_strahler_single_segment_is_order_one():
    net = network_from_edges([0.0, 0.0], [0.0, 1.0], [(0, 1, 1.0, 1.0)], root=0)
    assert m.strahler_orders(net).tolist() == [1]
    assert m.strahler_summary(net)["max_order"] == 1


def test_fractal_tree_morphometry_is_physiological():
    net = arterial_tree(Parameters(seed=0))
    s = m.strahler_summary(net)
    assert s["max_order"] >= 3
    assert 1.5 < s["bifurcation_ratio"] < 5.0        # typical vascular R_B
    assert s["diameter_ratio"] > 1.0                 # higher-order vessels are thicker
    # segment counts fall monotonically toward the root
    ks = sorted(s["counts"])
    assert all(s["counts"][a] >= s["counts"][b] for a, b in pairwise(ks))
    # every segment received a positive order
    assert np.all(m.strahler_orders(net) > 0)
