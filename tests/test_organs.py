"""Organ-shaped tissue domains: every shape is one connected region (a split mask
makes the Darcy solve singular), sits inside its window, and trees solve in it."""

import numpy as np
import pytest
from scipy.ndimage import label

from perfusion import organs, treegen
from perfusion.core.geometry import rasterize_mask

D2 = (-70.0, 70.0, 0.0, 160.0)
D3 = (0.0, 60.0, 0.0, 60.0, 0.0, 60.0)


@pytest.mark.parametrize("name", list(organs.SHAPES_2D))
@pytest.mark.parametrize("n", [30, 60, 140])
def test_2d_organs_are_connected_and_inside(name, n):
    poly = np.array(organs.polygon(name, D2))
    assert poly[:, 0].min() >= D2[0]
    assert poly[:, 0].max() <= D2[1]
    assert poly[:, 1].min() >= D2[2]
    assert poly[:, 1].max() <= D2[3]
    mask = rasterize_mask(n, int(n * 8 / 7), D2, [poly.tolist()])
    assert label(mask)[1] == 1


@pytest.mark.parametrize("name", list(organs.SHAPES_3D))
@pytest.mark.parametrize("n", [12, 22, 32])
def test_3d_organs_are_connected(name, n):
    mask = organs.mask_3d(name, n, n, n, D3)
    assert mask.sum() > 0.02 * n**3
    assert label(mask)[1] == 1


def test_organs_keep_proportions_but_generic_shapes_stretch():
    wide = (0.0, 200.0, 0.0, 100.0)
    heart = np.array(organs.polygon("heart", wide))
    ellipse = np.array(organs.polygon("ellipse", wide))
    assert np.ptp(ellipse[:, 0]) / np.ptp(ellipse[:, 1]) == pytest.approx(2.0, rel=0.05)
    assert np.ptp(heart[:, 0]) / np.ptp(heart[:, 1]) < 1.4      # not stretched 2:1


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="unknown 2-D shape"):
        organs.polygon("spleen-ish", D2)
    with pytest.raises(ValueError, match="unknown 3-D shape"):
        organs.mask_3d("spleen-ish", 8, 8, 8, D3)


@pytest.mark.parametrize("name", list(organs.SHAPES_3D))
def test_trees_grow_and_solve_in_every_3d_organ(name):
    spec = treegen.GenSpec(dim=3, shape=name, nx=16, fill="cco", n_terminals=12,
                           generations=3, length_to_radius=5,
                           artery=treegen.TreeSpec((30, 30, 4), direction=(0, 0, 1), radius=1.5),
                           vein=treegen.TreeSpec((30, 30, 56), direction=(0, 0, -1), radius=1.8))
    art, ven, mask, dom, _ = treegen.build(spec)
    res = treegen.solve(spec, art, ven, mask, dom)
    q = res.solution.terminal_flux
    assert q[q > 0].sum() > 0
    assert q[q > 0].sum() == pytest.approx(-q[q < 0].sum(), rel=1e-6)


def test_extent_scales_the_domain_and_the_shape():
    small = treegen.GenSpec(shape="liver", fill="none")
    big = treegen.GenSpec(shape="liver", fill="none", extent=3.0)
    (_, d1, o1), (_, d3, o3) = treegen.tissue(small), treegen.tissue(big)
    assert d3[1] == pytest.approx(3 * d1[1])
    assert d3[3] == pytest.approx(3 * d1[3])
    assert np.ptp(np.array(o3)[:, 1]) == pytest.approx(3 * np.ptp(np.array(o1)[:, 1]))


@pytest.mark.parametrize("name", list(organs.SHAPES_2D))
def test_2d_anatomical_roots_lead_into_the_organ(name):
    from perfusion.core.geometry import _points_in_polygon
    poly = organs.polygon(name, D2)
    r = organs.roots(name, 2, D2)
    for kind in ("artery", "vein"):
        (x, y), a = r[kind]["root"], np.radians(r[kind]["angle"])
        steps = np.arange(0, 30, 2.0)
        inside = _points_in_polygon(x + steps * np.cos(a), y + steps * np.sin(a), poly)
        assert inside.any(), f"{name} {kind} trunk never enters the organ"


@pytest.mark.parametrize("name", list(organs.SHAPES_3D))
def test_3d_anatomical_roots_lead_into_the_organ(name):
    n, h = 30, 60 / 30
    mask = organs.mask_3d(name, n, n, n, D3)
    r = organs.roots(name, 3, D3)
    for kind in ("artery", "vein"):
        p, d = np.array(r[kind]["root"]), np.array(r[kind]["direction"])
        assert np.linalg.norm(d) == pytest.approx(1.0)
        idx = [np.clip(((p + s * d) / h).astype(int), 0, n - 1) for s in np.arange(0, 20, 1.0)]
        assert any(mask[tuple(i)] for i in idx), f"{name} {kind} trunk never enters the organ"
