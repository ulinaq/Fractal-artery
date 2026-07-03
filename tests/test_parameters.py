"""Tests for model parameters and bifurcation geometry."""

import numpy as np
import pytest

from perfusion.core.parameters import Parameters, bifurcation_geometry


def test_symmetric_bifurcation():
    # gamma = 1 -> identical children; delta = beta = 2^(-1/xi).
    phi, ratio = bifurcation_geometry(1.0, 3.0)
    assert ratio[0] == pytest.approx(ratio[1])
    assert ratio[0] == pytest.approx(0.5 ** (1 / 3), rel=1e-9)
    assert phi[0] == pytest.approx(phi[1])


def test_murray_law_radii_cube():
    # For xi = 3, r0^3 = r1^3 + r2^3 should hold for the child ratios.
    _, ratio = bifurcation_geometry(0.6, 3.0)
    assert ratio[0] ** 3 + ratio[1] ** 3 == pytest.approx(1.0, rel=1e-9)


def test_angles_in_range():
    phi, _ = bifurcation_geometry(0.4, 2.5)
    assert np.all((phi >= 0) & (phi <= np.pi))


def test_parameters_derived_fields():
    prm = Parameters(r0=2.0, r_min_frac=0.15, rv_scale=1.2, seed=1)
    assert prm.r_min == pytest.approx(0.3)
    assert prm.rv0 == pytest.approx(2.4)
    assert prm.rv_min == pytest.approx(1.2 * 0.3)
    # generation tables: 6 arterial (1 + 5), 6 venous
    assert prm.phi.shape == (6, 2)
    assert prm.vphi.shape == (6, 2)
    assert prm.ratio.shape == (6, 2)


def test_seed_reproducibility():
    a = Parameters(seed=42)
    b = Parameters(seed=42)
    assert np.allclose(a.ratio, b.ratio)
    assert np.allclose(a.vphi, b.vphi)
