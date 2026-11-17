"""Property-based invariants: mass conservation must hold across the parameter space."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from perfusion import Grid, Parameters, run_hybrid, run_multicompartment  # noqa: E402

DOM = (-70, 70, 0, 160)
_SETTINGS = settings(max_examples=25, deadline=None,
                     suppress_health_check=[HealthCheck.function_scoped_fixture])


@_SETTINGS
@given(
    p_in=st.floats(min_value=2.0, max_value=40.0),
    mu=st.floats(min_value=0.5, max_value=3.0),
    r_min_frac=st.floats(min_value=0.35, max_value=0.7),
    perm=st.floats(min_value=0.2, max_value=5.0),
)
def test_hybrid_conserves_mass_everywhere(p_in, mu, r_min_frac, perm):
    grid = Grid(nx=40, ny=48, lx=140, ly=160, permeability=perm)
    prm = Parameters(p_in=p_in, p_out=0.0, mu=mu, r_min_frac=r_min_frac, seed=0)
    res = run_hybrid(prm, grid, domain_limit=DOM)
    tf = res.solution.terminal_flux
    inflow, outflow = tf[tf > 0].sum(), -tf[tf < 0].sum()
    # arterial inflow == venous outflow for any admissible parameters
    assert inflow == pytest.approx(outflow, rel=1e-6, abs=1e-9)
    assert np.isfinite(res.pressure).any()


@_SETTINGS
@given(
    p_in=st.floats(min_value=2.0, max_value=40.0),
    beta=st.floats(min_value=0.1, max_value=5.0),
    n=st.integers(min_value=2, max_value=4),
)
def test_multicompartment_throughflow_matches_inflow(p_in, beta, n):
    grid = Grid(nx=36, ny=44, lx=140, ly=160)
    prm = Parameters(p_in=p_in, p_out=0.0, r_min_frac=0.5, seed=0)
    res = run_multicompartment(prm, grid, permeabilities=tuple([1.0] * n),
                               betas=tuple([beta] * (n - 1)), domain_limit=DOM)
    tf = res.solution.terminal_flux
    inflow = tf[tf > 0].sum()
    # the first inter-compartment exchange carries the whole delivered flow
    assert np.nansum(res.perfusion[0]) == pytest.approx(inflow, rel=2e-3)
