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


def test_terminals_map_to_the_cell_that_contains_them():
    # Cell i spans [xmin + i h, xmin + (i+1) h): arteries used ceil (a MATLAB
    # 1-based leftover) and landed one cell up and right of the veins.
    from perfusion import Grid
    from perfusion.core.geometry import network_from_polylines
    from perfusion.core.pipeline import _terminal_cells

    grid = Grid(nx=10, ny=10, lx=10, ly=10)
    art = network_from_polylines([[(0.5, 0.2), (3.5, 6.5)]], radii=1.0, root_xy=(0.5, 0.2))
    ven = network_from_polylines([[(9.5, 9.8), (3.5, 6.5)]], radii=1.0, root_xy=(9.5, 9.8))
    a, v = _terminal_cells(grid, art, ven, (0, 10, 0, 10), strict=True)
    assert a == grid.sub2ind(3, 6)
    assert v != a                             # coincident -> spread to a neighbour
    # the last column is inside the grid, so strict mode must accept it
    art_edge = network_from_polylines([[(0.5, 0.2), (9.9, 9.9)]], radii=1.0,
                                      root_xy=(0.5, 0.2))
    (a_edge, _) = _terminal_cells(grid, art_edge, ven, (0, 10, 0, 10), strict=True)
    assert a_edge in (grid.sub2ind(9, 9), grid.sub2ind(8, 9), grid.sub2ind(9, 8))


def test_elasticity_and_phase_separation_combine(grid):
    # They were an if/elif pair: asking for both silently dropped phase separation.
    import warnings

    from perfusion import Parameters, run_hybrid

    prm = Parameters(seed=0, r_min_frac=0.5, compliance=0.02, radius_to_um=20.0,
                     fahraeus_lindqvist=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        el = run_hybrid(prm, grid, elasticity=True)
        both = run_hybrid(prm, grid, elasticity=True, phase_separation=True)
    assert not np.isclose(el.solution.q_a[0], both.solution.q_a[0], rtol=1e-6)
    assert both.solution.q_a[0] == pytest.approx(both.solution.q_v[0], rel=1e-6)
