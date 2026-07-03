"""Tests for flux magnitude and time-of-flight."""

import numpy as np

from perfusion import timeofflight


def test_flux_magnitude_shape_and_nonneg(flow_field):
    grid, fx, fy, _, _ = flow_field
    mag = timeofflight.flux_magnitude(grid, fx, fy)
    assert mag.shape == (grid.nx, grid.ny)
    assert np.all(mag >= 0)


def test_time_of_flight_finite_and_positive(flow_field):
    grid, fx, fy, sources, flux = flow_field
    tof = timeofflight.time_of_flight(grid, fx, fy, sources, flux)
    assert tof.shape == (grid.nx, grid.ny)
    finite = tof[np.isfinite(tof)]
    assert finite.size > 0
    assert np.all(finite >= 0)


def test_time_of_flight_grows_downstream(flow_field):
    # Flow runs from the (10,10) source toward the (30,30) sink, so cells near
    # the source are reached sooner than cells far downstream.
    grid, fx, fy, sources, flux = flow_field
    tof = timeofflight.time_of_flight(grid, fx, fy, sources, flux)
    near = tof[12, 12]
    far = tof[28, 28]
    assert np.isfinite(near)
    assert np.isfinite(far)
    assert far > near
