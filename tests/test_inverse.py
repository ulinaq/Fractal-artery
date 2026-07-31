"""Inverse calibration — recover planted model inputs from a perfusion measurement."""

import numpy as np
import pytest

from perfusion import Grid, Parameters, inverse
from perfusion.core.pipeline import run_hybrid

DOM = (-70, 70, 0, 160)


@pytest.fixture(scope="module")
def grid():
    return Grid(nx=50, ny=60, lx=140, ly=160)


def _inflow(result):
    tf = result.solution.terminal_flux
    return float(tf[tf > 0].sum())


def test_recover_inlet_pressure(grid):
    base = {"p_out": 0.0, "r_min_frac": 0.4, "seed": 0}
    target = _inflow(run_hybrid(Parameters(p_in=13.0, **base), grid, domain_limit=DOM))
    r = inverse.calibrate(target, "p_in", grid, (1.0, 40.0), base=base, domain_limit=DOM)
    assert r.value == pytest.approx(13.0, abs=0.3)
    assert r.achieved == pytest.approx(target, rel=1e-3)
    assert r.n_eval >= 3


def test_recover_permeability_from_mean_flux(grid):
    base = {"p_in": 10.0, "p_out": 0.0, "r_min_frac": 0.4, "seed": 0}
    planted = Grid(nx=50, ny=60, lx=140, ly=160, permeability=2.5)
    res = run_hybrid(Parameters(**base), planted, domain_limit=DOM)
    target = float(res.flux[np.isfinite(res.flux)].mean())
    r = inverse.calibrate(target, "permeability", grid, (0.5, 8.0), qoi="flux_mean",
                          base=base, domain_limit=DOM)
    assert r.value == pytest.approx(2.5, rel=0.1)


def test_unknown_qoi_and_param_raise(grid):
    with pytest.raises(ValueError, match="unknown qoi"):
        inverse.calibrate(1.0, "p_in", grid, (1.0, 20.0), qoi="nope", domain_limit=DOM)
    with pytest.raises(ValueError, match="unknown param"):
        inverse.calibrate(1.0, "nope", grid, (1.0, 20.0), domain_limit=DOM)
