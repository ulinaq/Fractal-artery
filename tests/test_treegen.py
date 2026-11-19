"""Tree-explorer growth (``perfusion.treegen``): grid-independent growth, lesions with
their own permeability / vessel density, and colonisation from an overshooting
skeleton."""

from dataclasses import replace

import numpy as np
import pytest

from perfusion import treegen


def _spec(**kw):
    base = {"dim": 2, "shape": "kidney", "fill": "colonization", "nx": 40, "ny": 46}
    return treegen.GenSpec(**{**base, **kw})


@pytest.mark.parametrize("fill", ["colonization", "cco"])
def test_trees_do_not_depend_on_the_solve_grid(fill):
    """Growth samples a fixed raster, so the optimiser's coarse search grid grows the
    very same trees as the user's grid."""
    a1, v1, *_ = treegen.build(_spec(fill=fill, nx=30, ny=34))
    a2, v2, *_ = treegen.build(_spec(fill=fill, nx=90, ny=103))
    for n1, n2 in ((a1, a2), (v1, v2)):
        assert np.allclose(n1.positions(), n2.positions())
        assert np.array_equal(n1.seg_start, n2.seg_start)


def _lesion(density=1.0, permeability=0.1, center=(0.0, 80.0), radius=25.0):
    return {"center": list(center), "radius": radius, "permeability": permeability,
            "density": density}


def _inside(spec, net):
    le = spec.lesions[0]
    d = np.linalg.norm(net.positions(net.terminal_nodes) - le["center"], axis=1)
    return int((d <= le["radius"]).sum())


def test_lesion_vessel_density_steers_growth():
    counts = []
    for density in (0.0, 1.0, 5.0):
        spec = _spec(lesions=[_lesion(density)])
        art, ven, *_ = treegen.build(spec)
        counts.append(_inside(spec, art) + _inside(spec, ven))
    assert counts[0] < counts[1] < counts[2]          # avascular < healthy < angiogenic
    assert counts[0] < 0.25 * counts[1]           # only skeleton tips that end there


def test_lesion_permeability_lowers_its_perfusion_and_is_left_out_of_the_score():
    spec = _spec(fill="cco", n_terminals=30, lesions=[_lesion(permeability=1.0)])
    art, ven, mask, domain, _ = treegen.build(spec)
    same = treegen.lesion_stats(spec, treegen.solve(spec, art, ven, mask, domain))
    low = replace(spec, lesions=[_lesion(permeability=0.05)])
    res = treegen.solve(low, art, ven, mask, domain)   # the same trees, low-K lesion
    stats = treegen.lesion_stats(low, res)
    assert stats["cells"] > 0
    assert stats["healthy_cells"] + stats["cells"] == int(mask.sum())
    assert stats["flux_ratio"] < same["flux_ratio"]
    # the homogeneity objective scores the healthy tissue only
    inside = treegen.lesion_mask(low, mask, domain)
    full, healthy = treegen.homogeneity(res), treegen.homogeneity(res, exclude=inside)
    flux = res.flux[mask & ~inside]
    assert healthy["flux_std"] == pytest.approx(np.nanstd(flux))
    assert healthy["flux_std"] != pytest.approx(full["flux_std"])
    assert treegen.lesion_stats(_spec(), res) is None              # no lesions


def test_lesion_input_is_checked():
    with pytest.raises(ValueError, match="2 values"):
        treegen.build(_spec(lesions=[_lesion(center=(0, 1, 2))]))
    with pytest.raises(ValueError, match="no vascularised tissue"):
        treegen.build(_spec(lesions=[_lesion(density=0.0, radius=500.0)]))


def _spec_3d(shape, **kw):
    """A 3-D setup from the organ's anatomical roots (as the MCP tools build it)."""
    from perfusion import organs

    roots = organs.roots(shape, 3, treegen.domain_for(3, 1.0))
    art, ven = roots["artery"], roots["vein"]
    return treegen.GenSpec(
        dim=3, shape=shape, fill="colonization", nx=16,
        artery=treegen.TreeSpec(tuple(art["root"]), 90.0, tuple(art["direction"])),
        vein=treegen.TreeSpec(tuple(ven["root"]), 90.0, tuple(ven["direction"]),
                              radius=2.4, gamma=1.2), **kw)


def test_colonisation_grows_from_an_overshooting_3d_skeleton():
    """The kidney's 3-D skeleton overshoots the organ (every tip outside reach of the
    tissue); colonisation then sprouts from the trunk nodes that do reach it."""
    spec = _spec_3d("kidney")
    skeleton = treegen.build(replace(spec, fill="none"))[0]
    art = treegen.build(spec)[0]
    assert art.terminal_nodes.size > 3 * skeleton.terminal_nodes.size


def test_3d_lesion_density_steers_growth():
    counts = []
    for density in (0.0, 5.0):
        spec = _spec_3d("ellipsoid", lesions=[{"center": [30, 30, 30], "radius": 12,
                                               "permeability": 0.1, "density": density}])
        art, ven, *_ = treegen.build(spec)
        counts.append(_inside(spec, art) + _inside(spec, ven))
    assert counts[0] < counts[1]
