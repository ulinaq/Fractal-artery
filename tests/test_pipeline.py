"""Tests for the end-to-end hybrid pipeline, including a regression lock."""

import numpy as np
import pytest


def test_hybrid_result_fields(hybrid_result):
    res = hybrid_result
    assert res.pressure.shape == (res.grid.nx, res.grid.ny)
    assert res.flux.shape == (res.grid.nx, res.grid.ny)
    assert res.tof.shape == (res.grid.nx, res.grid.ny)
    assert np.isfinite(res.pressure).all()
    assert res.solution.q_a[0] > 0


def test_hybrid_mass_balance(hybrid_result):
    sol = hybrid_result.solution
    assert sol.q_a[0] == pytest.approx(sol.q_v[0], rel=1e-6)


def test_pressure_gradient_direction(hybrid_result):
    res = hybrid_result
    art, ven = res.art, res.ven
    assert res.solution.p_a[art.terminal_nodes].mean() > \
        res.solution.p_v[ven.terminal_nodes].mean()


def test_regression_against_reference(hybrid_result, reference_hybrid):
    # Numerics must stay locked to the committed reference (generate_data.py).
    res = hybrid_result
    ref = reference_hybrid
    assert res.art.n_nodes == int(ref["n_art_nodes"])
    assert res.ven.n_nodes == int(ref["n_ven_nodes"])
    assert res.solution.q_a[0] == pytest.approx(float(ref["q_a0"]), rel=1e-6)
    assert res.pressure.min() == pytest.approx(float(ref["pressure_min"]), rel=1e-6)
    assert res.pressure.max() == pytest.approx(float(ref["pressure_max"]), rel=1e-6)
    assert res.flux.std() == pytest.approx(float(ref["flux_std"]), rel=1e-6)
    tof = res.tof[np.isfinite(res.tof) & (res.tof > 0)]
    assert tof.std() == pytest.approx(float(ref["tof_std"]), rel=1e-6)


def test_nonlinear_pipeline_runs(small_params, grid):
    from perfusion import run_hybrid

    res = run_hybrid(small_params, grid, nonlinear=True)
    assert res.solution.q_a[0] == pytest.approx(res.solution.q_v[0], rel=1e-5)


def test_strict_domain_rejects_offgrid_tree(grid):
    from perfusion import Parameters, run_hybrid

    prm = Parameters(a_init=(1e6, 0.0), seed=0)   # root far outside the grid
    with pytest.raises(ValueError, match="outside the Darcy grid"):
        run_hybrid(prm, grid, strict_domain=True)


def test_fahraeus_lindqvist_changes_flow(grid):
    # Radius-dependent viscosity raises small-vessel resistance -> different flow.
    from perfusion import Parameters, run_hybrid

    base = run_hybrid(Parameters(seed=0), grid)
    fl = run_hybrid(Parameters(seed=0, fahraeus_lindqvist=True, radius_to_um=20.0),
                    grid)
    assert not np.isclose(base.solution.q_a[0], fl.solution.q_a[0])
    assert fl.solution.q_a[0] == pytest.approx(fl.solution.q_v[0], rel=1e-6)


def test_terminal_cells_are_unique(hybrid_result):
    # #8: coincident terminals get spread so every Darcy source is distinct.
    from perfusion.core.pipeline import _terminal_cells

    res = hybrid_result
    cells = _terminal_cells(res.grid, res.art, res.ven, (-70, 70, 0, 160))
    assert len(np.unique(cells)) == len(cells)


def test_spread_coincident_resolves_duplicates(grid):
    from perfusion.core.pipeline import _spread_coincident

    cells = np.array([100, 100, 100, 205, 205])
    spread = _spread_coincident(grid, cells)
    assert len(np.unique(spread)) == len(spread)
