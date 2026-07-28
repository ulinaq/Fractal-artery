"""Multi-compartment Darcy bed (core.multicompartment + pipeline.run_multicompartment)."""

import numpy as np
import pytest

from perfusion import Grid, Parameters, run_hybrid, run_multicompartment


@pytest.fixture(scope="module")
def grid():
    return Grid(nx=50, ny=60, lx=140, ly=160)


@pytest.fixture(scope="module")
def prm():
    return Parameters(r_min_frac=0.4, seed=0)


def test_single_compartment_reduces_to_hybrid(prm, grid):
    # N = 1 with no exchange must reproduce the single-bed hybrid exactly
    h = run_hybrid(prm, grid, domain_limit=(-70, 70, 0, 160))
    m = run_multicompartment(prm, grid, permeabilities=(1.0,), betas=(),
                             domain_limit=(-70, 70, 0, 160))
    assert np.allclose(h.solution.terminal_flux, m.solution.terminal_flux, atol=1e-9)
    assert np.allclose(np.nan_to_num(h.pressure), np.nan_to_num(m.compartments[0]), atol=1e-9)


def test_two_compartment_conserves_and_percolates(prm, grid):
    m = run_multicompartment(prm, grid, permeabilities=(1.0, 0.5), betas=(1.0,),
                             domain_limit=(-70, 70, 0, 160))
    tf = m.solution.terminal_flux
    inflow, outflow = tf[tf > 0].sum(), -tf[tf < 0].sum()
    assert inflow == pytest.approx(outflow, rel=1e-6)                     # mass conserved
    # all the through-flow crosses the single compartment interface (capillary perfusion)
    assert np.nansum(m.perfusion[0]) == pytest.approx(outflow, rel=1e-3)
    # arteriole sits above the venule, so exchange flows downstream on average
    assert np.nanmean(m.compartments[0]) > np.nanmean(m.compartments[1])
    assert np.nanmean(m.perfusion[0]) > 0.0


def test_three_compartment_interfaces_match(prm, grid):
    m = run_multicompartment(prm, grid, permeabilities=(1.0, 0.7, 0.4), betas=(1.0, 1.0),
                             domain_limit=(-70, 70, 0, 160))
    assert m.compartments.shape == (3, grid.nx, grid.ny)
    assert m.perfusion.shape == (2, grid.nx, grid.ny)
    # steady state: the same flow crosses arteriole→capillary and capillary→venule
    assert np.nansum(m.perfusion[0]) == pytest.approx(np.nansum(m.perfusion[1]), rel=1e-3)


def test_beta_length_validated(prm, grid):
    with pytest.raises(ValueError, match="betas must have length"):
        run_multicompartment(prm, grid, permeabilities=(1.0, 0.5), betas=(1.0, 1.0))
