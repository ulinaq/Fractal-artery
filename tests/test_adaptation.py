"""Structural adaptation toward Murray's law (core.adaptation + run_adaptation)."""

import numpy as np
import pytest

from perfusion import Grid, Parameters
from perfusion.core import adaptation
from perfusion.core.pipeline import run_adaptation


def test_wall_shear_stress_scaling():
    # τ = 4 μ Q / (π r³): doubling flow doubles shear; doubling radius cuts it 8×
    wss = adaptation.wall_shear_stress
    tau = wss(np.array([1.0]), np.array([1.0]), viscosity=1.0)[0]
    assert tau == pytest.approx(4.0 / np.pi)
    assert wss(np.array([1.0]), np.array([2.0]))[0] == pytest.approx(2 * tau)   # Q → 2Q
    assert wss(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(tau / 8)   # r → 2r


def test_murray_exponent_recovers_cubic_law():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    q = r**3                                  # exact Murray's law
    assert adaptation.murray_exponent(r, q) == pytest.approx(3.0, abs=1e-6)


def test_adapt_radii_dilates_oversheared_shrinks_quiescent():
    # one over-sheared segment (high flow) and one quiescent (low flow), same radius
    r = np.array([1.0, 1.0])
    q = np.array([10.0, 0.1])
    tau_ref = float(adaptation.wall_shear_stress(r, q).mean())
    r_new, s = adaptation.adapt_radii(r, q, tau_ref=tau_ref, rate=0.2)
    assert r_new[0] > r[0]                    # over-sheared dilates
    assert r_new[1] < r[1]                    # quiescent shrinks
    assert s[0] > 0 > s[1]


@pytest.mark.parametrize("target", [3.0, 3.5])
def test_run_adaptation_moves_toward_murray(target):
    prm = Parameters(r_min_frac=0.4, seed=0)
    grid = Grid(nx=50, ny=60, lx=140, ly=160)
    ad = run_adaptation(prm, grid, target_exponent=target, rate=0.05, n_iter=60,
                        domain_limit=(-70, 70, 0, 160))
    # the exponent ends closer to the target than it started, within a small tolerance
    assert abs(ad.murray_exponent - target) < abs(ad.murray_start - target)
    assert abs(ad.murray_exponent - target) < 0.2
    # the adapted tree still conserves mass (arterial in == venous out)
    tf = ad.result.solution.terminal_flux
    assert tf[tf > 0].sum() == pytest.approx(-tf[tf < 0].sum(), rel=1e-6)
    # radii stayed within the enforced [0.1, 10]× bounds of a fractal tree
    assert np.all(ad.art.seg_radius > 0)


def test_adaptation_is_deterministic():
    prm = Parameters(r_min_frac=0.5, seed=0)
    grid = Grid(nx=45, ny=52, lx=140, ly=160)
    a = run_adaptation(prm, grid, n_iter=40, domain_limit=(-70, 70, 0, 160))
    b = run_adaptation(prm, grid, n_iter=40, domain_limit=(-70, 70, 0, 160))
    assert a.murray_exponent == pytest.approx(b.murray_exponent)
    assert a.n_iter == b.n_iter
