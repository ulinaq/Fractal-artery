"""Tests for passive tracer transport (Papers B/C)."""

import numpy as np
import pytest

from perfusion import tkm, transport


def test_build_advection_split(flow_field):
    grid, fx, fy, sources, flux = flow_field
    A, inject, extract, pore_volume = transport.build_advection(
        grid, fx, fy, sources, flux)
    assert A.shape == (grid.n_cells, grid.n_cells)
    # arterial terminal (flux > 0) injects, venous terminal (flux < 0) extracts
    assert inject.sum() > 0
    assert extract.sum() > 0
    assert np.all(pore_volume > 0)


def test_transport_forward_snapshots(flow_field):
    grid, fx, fy, sources, flux = flow_field
    t = np.arange(0.0, 30.0, 0.1)
    aif = tkm.gamma_variate_aif(t, t0=2.0, A=3.0, B=1.0)
    frames = transport.transport_forward(grid, fx, fy, sources, flux, aif, dt=0.1)
    assert frames.shape == (len(aif), grid.nx, grid.ny)
    assert np.all(frames >= 0)                    # concentrations stay non-negative
    assert frames[0].sum() == 0.0                 # empty before any inflow
    assert frames[-1].sum() > frames[10].sum()    # tracer accumulates over time


def test_transport_tracer_reaches_tissue(flow_field):
    grid, fx, fy, sources, flux = flow_field
    t = np.arange(0.0, 40.0, 0.1)
    aif = tkm.gamma_variate_aif(t, t0=2.0)
    frames = transport.transport_forward(grid, fx, fy, sources, flux, aif, dt=0.1)
    # the injection region (around cell 10,10) sees tracer
    assert frames[:, 10, 10].max() > 0


def test_cfl_limit_positive(flow_field):
    grid, fx, fy, sources, flux = flow_field
    assert transport.cfl_limit(grid, fx, fy, sources, flux) > 0


def test_implicit_stable_at_large_dt(flow_field):
    # Implicit Euler (theta=1) stays bounded well beyond the CFL limit.
    grid, fx, fy, sources, flux = flow_field
    limit = transport.cfl_limit(grid, fx, fy, sources, flux)
    aif = np.ones(50)
    frames = transport.transport_forward(grid, fx, fy, sources, flux, aif,
                                         dt=50 * limit, theta=1.0)
    assert np.all(np.isfinite(frames))
    assert frames.max() < 1e3               # no blow-up


def test_explicit_cfl_warning(flow_field):
    grid, fx, fy, sources, flux = flow_field
    limit = transport.cfl_limit(grid, fx, fy, sources, flux)
    aif = np.ones(5)
    with pytest.warns(RuntimeWarning):
        transport.transport_forward(grid, fx, fy, sources, flux, aif,
                                    dt=10 * limit, theta=0.0)


def test_explicit_implicit_agree_small_dt(flow_field):
    grid, fx, fy, sources, flux = flow_field
    limit = transport.cfl_limit(grid, fx, fy, sources, flux)
    t = np.arange(0.0, 5.0, 0.2 * limit)
    aif = tkm.gamma_variate_aif(t, t0=1.0)
    expl = transport.transport_forward(grid, fx, fy, sources, flux, aif,
                                       dt=0.2 * limit, theta=0.0)
    impl = transport.transport_forward(grid, fx, fy, sources, flux, aif,
                                       dt=0.2 * limit, theta=1.0)
    assert np.allclose(expl[-1], impl[-1], atol=1e-3)
