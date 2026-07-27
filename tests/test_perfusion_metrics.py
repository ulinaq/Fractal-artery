"""Transit-time distribution metrics (MTT / CTH / RD) and the oxygen-extraction
heterogeneity penalty."""

import numpy as np
import pytest

from perfusion.core import perfusion_metrics as pm


def test_mtt_cth_rd_of_known_values():
    tof = np.array([1.0, 2.0, 3.0])
    m = pm.transit_metrics(tof)
    assert m.mtt == pytest.approx(2.0)
    assert m.cth == pytest.approx(np.std(tof))            # population std
    assert m.rd == pytest.approx(m.cth / m.mtt)
    assert m.n == 3


def test_nans_are_dropped_and_weights_apply():
    tof = np.array([[1.0, np.nan], [3.0, 5.0]])
    # unweighted mean over finite {1,3,5} = 3
    assert pm.transit_metrics(tof).mtt == pytest.approx(3.0)
    assert pm.transit_metrics(tof).n == 3
    # weight the finite cells (nan cell's weight is ignored)
    w = np.array([[10.0, 100.0], [1.0, 1.0]])
    wm = pm.transit_metrics(tof, weights=w)
    assert wm.mtt == pytest.approx((1 * 10 + 3 * 1 + 5 * 1) / 12)


def test_empty_field_is_zero():
    m = pm.transit_metrics(np.full((3, 3), np.nan))
    assert (m.mtt, m.cth, m.rd, m.n) == (0.0, 0.0, 0.0, 0)


def test_histogram_is_a_normalised_density():
    rng = np.random.default_rng(0)
    tof = rng.gamma(shape=3.0, scale=1.0, size=5000)
    centres, density = pm.transit_histogram(tof, bins=50)
    width = centres[1] - centres[0]
    assert np.sum(density) * width == pytest.approx(1.0, rel=1e-6)   # integrates to 1
    assert len(centres) == len(density) == 50


def test_heterogeneity_lowers_oxygen_extraction():
    # same MTT (=2), different spread: the concave Kety extraction must penalise
    # the heterogeneous distribution (Jensen's inequality).
    homogeneous = np.array([2.0, 2.0, 2.0, 2.0])
    heterogeneous = np.array([0.5, 0.5, 3.5, 3.5])
    assert pm.transit_metrics(homogeneous).mtt == pytest.approx(2.0)
    assert pm.transit_metrics(heterogeneous).mtt == pytest.approx(2.0)
    assert pm.transit_metrics(heterogeneous).cth > pm.transit_metrics(homogeneous).cth
    oef_hom = pm.oxygen_extraction_fraction(homogeneous, k=1.0)
    oef_het = pm.oxygen_extraction_fraction(heterogeneous, k=1.0)
    assert 0.0 < oef_het < oef_hom < 1.0
