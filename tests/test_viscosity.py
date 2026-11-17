"""Tests for the Fahraeus-Lindqvist apparent viscosity."""

import numpy as np
import pytest

from perfusion import viscosity


def test_positive_and_scalar_shape():
    assert viscosity.apparent_viscosity(50.0) > 0
    out = viscosity.apparent_viscosity(np.array([20.0, 100.0, 300.0]))
    assert out.shape == (3,)
    assert np.all(out > 0)


def test_fahraeus_lindqvist_is_u_shaped():
    # The apparent viscosity has a minimum around ~50 um: it rises steeply toward
    # capillary (red-cell) size and gently again for large vessels.
    d = np.array([8.0, 20.0, 50.0, 100.0, 200.0])
    mu = viscosity.apparent_viscosity(d)
    assert mu[0] > mu[2]              # capillary-size vessels are much more viscous
    assert mu[2] < mu[1]             # interior minimum near 50 um
    assert mu[2] < mu[3] < mu[4]     # gentle rise for larger vessels


def test_plasma_scaling_is_linear():
    d = np.array([15.0, 40.0])
    base = viscosity.apparent_viscosity(d, mu_plasma=1.0)
    scaled = viscosity.apparent_viscosity(d, mu_plasma=3.0)
    assert scaled == pytest.approx(3.0 * base)


def test_hematocrit_monotonic():
    # Higher hematocrit -> higher apparent viscosity.
    low = viscosity.apparent_viscosity(30.0, hematocrit=0.30)
    high = viscosity.apparent_viscosity(30.0, hematocrit=0.60)
    assert high > low
