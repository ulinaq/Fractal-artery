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


# -- multi-parameter DCE-MRI fit ----------------------------------------------- #
def _dce_problem(grid, p_in=14.0, porosity=0.3, n=40, art=None, ven=None, dom=DOM):
    """A synthetic DCE acquisition from the hybrid model: (times, tissue, aif) planted
    at (p_in, porosity), plus the fixed trees."""
    from dataclasses import replace

    from perfusion.core import tkm
    from perfusion.core.geometry import arterial_tree, venous_tree
    from perfusion.uq import _sample_solver

    if art is None:
        prm = Parameters(r_min_frac=0.5, seed=0)
        art, ven = arterial_tree(prm), venous_tree(prm)
    solve = _sample_solver(["p_in"], grid, dom, art, ven, None)
    truth = solve(Parameters(p_in=p_in, r_min_frac=0.5, seed=0),
                  replace(grid, porosity=porosity))
    tof = truth.tof[np.isfinite(truth.tof)]
    t_end = 1.2 * np.percentile(tof, 80) * porosity
    t = np.linspace(0.0, t_end, n)
    aif = tkm.gamma_variate_aif(t, t0=0.05 * t_end, A=3.0, B=0.05 * t_end)
    aif = aif / aif.max()
    return t, inverse.simulate_tissue_curve(truth, t, aif, porosity=porosity), aif, art, ven


def test_dce_fit_recovers_flow_and_blood_volume():
    """p_in (~ flow) and porosity (~ blood volume) from one tissue curve: exact on clean
    data, nearly uncorrelated, and flow solves are reused across porosity steps."""
    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    t, c, aif, art, ven = _dce_problem(grid)
    r = inverse.fit_dce(t, c, aif, grid, {"p_in": (1, 50), "porosity": (0.02, 1)},
                        base={"p_in": 10.0, "r_min_frac": 0.5, "seed": 0},
                        domain_limit=DOM, art=art, ven=ven)
    assert r.params["p_in"] == pytest.approx(14.0, rel=1e-4)
    assert r.params["porosity"] == pytest.approx(0.3, rel=1e-4)
    assert r.r2 == pytest.approx(1.0, abs=1e-9)
    assert abs(r.correlation[0][1]) < 0.9
    assert not r.warnings
    assert r.n_solve < r.n_eval                    # porosity-only steps skip the solve


def test_dce_fit_with_noise_gives_honest_errors():
    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    t, c, aif, art, ven = _dce_problem(grid)
    rng = np.random.default_rng(1)
    noisy = c + 0.02 * c.max() * rng.standard_normal(c.size)
    r = inverse.fit_dce(t, noisy, aif, grid, {"p_in": (1, 50), "porosity": (0.02, 1)},
                        base={"p_in": 10.0, "r_min_frac": 0.5, "seed": 0},
                        domain_limit=DOM, art=art, ven=ven)
    for name, truth in (("p_in", 14.0), ("porosity", 0.3)):
        assert r.params[name] == pytest.approx(truth, rel=0.1)
        assert 0 < r.std[name] < 0.2 * truth
        assert abs(r.params[name] - truth) < 4 * r.std[name]   # the error bar covers it
    assert r.r2 > 0.95


def test_dce_fit_flags_unidentifiable_pairs_and_bad_input():
    """p_in and mu only enter the linear model as their ratio → flagged."""
    grid = Grid(nx=30, ny=34, lx=140, ly=160)
    t, c, aif, art, ven = _dce_problem(grid, n=25)
    r = inverse.fit_dce(t, c, aif, grid, {"p_in": (1, 50), "mu": (0.1, 10)},
                        base={"r_min_frac": 0.5, "seed": 0, "p_in": 10.0},
                        start={"mu": 2.0}, domain_limit=DOM, art=art, ven=ven)
    assert any("strongly correlated" in w or "not identifiable" in w for w in r.warnings)
    with pytest.raises(ValueError, match="cannot fit"):
        inverse.fit_dce(t, c, aif, grid, {"hematocrit": (0.1, 0.6)}, art=art, ven=ven)
    with pytest.raises(ValueError, match="same length"):
        inverse.fit_dce(t, c[:-1], aif, grid, {"p_in": (1, 50)}, art=art, ven=ven)
    with pytest.raises(ValueError, match="increasing"):
        inverse.fit_dce(t[::-1], c, aif, grid, {"p_in": (1, 50)}, art=art, ven=ven)


def test_dce_fit_in_3d():
    from perfusion.core.darcy3d import Grid3D
    from perfusion.core.geometry import grow_tree_3d

    dom = (0.0, 40.0, 0.0, 40.0, 0.0, 40.0)
    grid = Grid3D(nx=10, ny=10, nz=10, lx=40, ly=40, lz=40)
    art = grow_tree_3d((10.0, 20.0, 2.0), (0, 0, 1), root_radius=2.0, min_radius=0.6,
                       length_to_radius=4.0, max_generations=3)
    ven = grow_tree_3d((30.0, 20.0, 2.0), (0, 0, 1), root_radius=2.4, min_radius=0.6,
                       length_to_radius=4.0, max_generations=3)
    t, c, aif, art, ven = _dce_problem(grid, p_in=12.0, porosity=0.4, n=30, art=art,
                                       ven=ven, dom=dom)
    r = inverse.fit_dce(t, c, aif, grid, {"p_in": (1, 50), "porosity": (0.02, 1)},
                        base={"p_in": 10.0}, domain_limit=dom, art=art, ven=ven)
    assert r.params["p_in"] == pytest.approx(12.0, rel=1e-3)
    assert r.params["porosity"] == pytest.approx(0.4, rel=1e-3)


def test_dce_fit_at_a_bound_keeps_the_other_error_bars():
    """A parameter pinned at its bound gets no standard error (and a warning), while
    the free one keeps a finite one."""
    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    t, c, aif, art, ven = _dce_problem(grid, porosity=1.0)
    r = inverse.fit_dce(t, c * 1.05, aif, grid, {"p_in": (1, 50), "porosity": (0.02, 1)},
                        base={"p_in": 10.0, "r_min_frac": 0.5, "seed": 0},
                        domain_limit=DOM, art=art, ven=ven)
    assert r.params["porosity"] == pytest.approx(1.0, rel=1e-3)
    assert r.std["porosity"] is None
    assert r.std["p_in"] is not None
    assert r.std["p_in"] > 0
    assert any("bound" in w for w in r.warnings)
