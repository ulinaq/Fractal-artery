"""Steady oxygen / solute transport + extraction (core.oxygen)."""

from itertools import pairwise

import numpy as np
import pytest

from perfusion import Grid, Parameters, run_hybrid
from perfusion.core import oxygen
from perfusion.core.transport import build_advection


@pytest.fixture(scope="module")
def result():
    prm = Parameters(r_min_frac=0.4, seed=0)
    grid = Grid(nx=60, ny=70, lx=140, ly=160)
    return run_hybrid(prm, grid, domain_limit=(-70, 70, 0, 160))


def test_solute_mass_is_conserved(result):
    # delivery = tissue consumption + venous washout, exactly (steady balance)
    ox = oxygen.oxygen_from_result(result, c_art=1.0, k=1.0)
    _, _inject, extract, _pv = build_advection(
        result.grid, result.fx, result.fy, result.sources, result.solution.terminal_flux)
    c = np.nan_to_num(ox.concentration.ravel(order="F"))
    venous = float((extract * c).sum())
    assert ox.delivery == pytest.approx(ox.consumption + venous, rel=1e-6, abs=1e-9)


def test_oef_monotone_and_bounded(result):
    oefs = [oxygen.oxygen_from_result(result, k=k).oef for k in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert all(0.0 < o < 1.0 for o in oefs)
    assert all(a < b for a, b in pairwise(oefs))   # more k → more extraction


def test_extraction_below_well_mixed_kety_limit(result):
    # transit-time heterogeneity lowers extraction below the well-mixed 1−e^{−k}
    k = 1.0
    ox = oxygen.oxygen_from_result(result, k=k)
    assert ox.oef < 1.0 - np.exp(-k)
    # saturation is a fraction of the arterial level, ≈1 near the feeds
    sat = ox.saturation
    assert np.nanmin(sat) >= 0.0
    assert np.nanmax(sat) <= 1.0 + 1e-9
    assert np.nanmax(sat) > 0.5


def test_diffusion_smooths_the_field(result):
    plain = oxygen.oxygen_from_result(result, k=1.0)
    diffused = oxygen.oxygen_from_result(result, k=1.0, diffusivity=1.0)
    assert np.nanstd(diffused.concentration) < np.nanstd(plain.concentration)
    # inactive cells stay NaN under the mask
    assert np.isnan(plain.concentration[~result.grid.mask]).all()


def test_requires_a_sink(result):
    with pytest.raises(ValueError, match="k>0 or diffusivity>0"):
        oxygen.oxygen_from_result(result, k=0.0, diffusivity=0.0)
