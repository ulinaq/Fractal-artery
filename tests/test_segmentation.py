"""Tests for image-to-network segmentation (needs the image extra)."""

import numpy as np
import pytest

pytest.importorskip("skimage")  # segmentation needs scikit-image

from perfusion import Grid, Parameters, run_hybrid
from perfusion import segmentation as seg
from perfusion.core.geometry import NodeRole


def _plus_image(size=60, cx=30, cy=30, arm=22, width=4):
    """A '+' of thick bright bars on a dark background (4 tips, 1 junction)."""
    img = np.zeros((size, size), dtype=np.uint8)
    h = width // 2
    img[cx - h:cx + h, cy - arm:cy + arm] = 255      # horizontal bar
    img[cx - arm:cx + arm, cy - h:cy + h] = 255      # vertical bar
    return img


def test_vessel_mask_threshold():
    img = _plus_image()
    mask = seg.vessel_mask(img, threshold=127)
    assert mask.dtype == bool
    assert mask.sum() == (img > 127).sum()


def test_segment_plus_topology():
    # A '+' skeletonises to 4 arms + 1 junction = 5 nodes, 4 segments; one arm
    # tip becomes the root, leaving 3 terminals.
    net = seg.segment_network(_plus_image(), min_size=5)
    assert net.n_nodes == 5
    assert net.n_segments == 4                        # a tree (nodes - 1)
    assert net.terminal_nodes.size == 3               # 4 tips minus the root tip
    assert net.bifurcation_nodes.size == 1
    assert np.all(net.seg_radius > 0)
    assert net.node_role[net.root_node] == NodeRole.ROOT


def test_segment_root_selection():
    # Root nearest a requested inlet coordinate.
    img = _plus_image(size=60, cx=30, cy=30, arm=22)
    net = seg.segment_network(img, root_xy=(8, 30), min_size=5)
    # the chosen root is the left-arm tip (x ~ 8)
    assert net.node_x[net.root_node] < 20


def test_segment_network_disconnected_keeps_largest():
    # Two separate plus shapes -> only the larger survives.
    img = np.zeros((80, 80), dtype=np.uint8)
    img[10:50, 28:32] = 255
    img[28:32, 10:50] = 255                            # big plus
    img[60:64, 60:70] = 255                            # tiny speckle
    net = seg.segment_network(img, min_size=5)
    assert net.n_nodes == 5                            # only the big plus survives
    assert net.terminal_nodes.size == 3                # speckle dropped


def test_hybrid_on_segmented_networks_conserves_mass():
    # Couple two segmented networks through a small Darcy bed.
    artery = seg.segment_network(_plus_image(60, cx=20, cy=30), min_size=5)
    vein = seg.segment_network(_plus_image(60, cx=42, cy=30), min_size=5)
    grid = Grid(nx=60, ny=60, lx=60, ly=60)
    res = run_hybrid(Parameters(), grid, domain_limit=(0, 60, 0, 60),
                     art=artery, ven=vein)
    tf = res.solution.terminal_flux
    assert abs(tf.sum()) < 1e-9                        # arterial in == venous out
    na = artery.terminal_nodes.size
    assert tf[:na].sum() > 0                           # arterial inflow positive


def test_load_frog_system():
    system = seg.load_frog_system(min_size=30)
    assert system.artery.terminal_nodes.size > 10
    assert system.vein.terminal_nodes.size > 10
    # arteries segment to a tree; veins contain loops (anastomoses)
    assert system.artery.n_segments == system.artery.n_nodes - 1
    assert system.vein.n_segments >= system.vein.n_nodes
    assert system.grid.mask.shape == system.mask.shape
