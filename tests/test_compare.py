"""Tests for run comparison (A/B): field differences and summary deltas."""

import numpy as np
import pytest

from perfusion.core.compare import compare_results, field_difference, field_stats, summary_delta


def test_field_difference_is_current_minus_baseline_on_shared_cells():
    base = np.array([[1.0, 2.0], [3.0, np.nan]])
    cur = np.array([[2.0, 2.0], [1.0, 5.0]])
    mask_cur = np.array([[True, False], [True, True]])
    d = field_difference(cur, base, mask_cur, None)
    assert d[0, 0] == 1.0
    assert np.isnan(d[0, 1])                 # inactive in the current run
    assert d[1, 0] == -2.0
    assert np.isnan(d[1, 1])                 # not finite in the baseline
    with pytest.raises(ValueError, match="grids differ"):
        field_difference(np.zeros((2, 3)), np.zeros((3, 2)))


def test_field_stats_and_identical_runs():
    rng = np.random.default_rng(0)
    f = rng.random((4, 5, 3))                # works in 3-D too
    same = field_stats(f, f)
    assert same["mean_abs_diff"] == 0.0
    assert same["rel_change"] == 0.0
    assert same["cells"] == f.size
    half = field_stats(0.5 * f, f)
    assert half["rel_change"] == pytest.approx(-0.5)
    assert half["max_abs_diff"] == pytest.approx(0.5 * f.max())


def test_summary_delta_keeps_shared_numbers():
    rows = summary_delta({"arterial_inflow": 1.5, "mtt": 4.0, "model": "hybrid",
                          "notes": ["x"], "flag": True},
                         {"arterial_inflow": 2.0, "mtt": 4.0, "model": "hybrid",
                          "flag": False, "only_base": 1.0})
    got = {r["key"]: r for r in rows}
    assert set(got) == {"arterial_inflow", "mtt"}
    assert got["arterial_inflow"]["diff"] == pytest.approx(-0.5)
    assert got["arterial_inflow"]["rel"] == pytest.approx(-0.25)
    assert got["mtt"]["diff"] == 0.0


def test_compare_results_across_grids_falls_back_to_the_summary():
    a = {"pressure": np.zeros((3, 3)), "flux": np.zeros((3, 3)), "tof": np.zeros((3, 3))}
    b = {"pressure": np.zeros((4, 4)), "flux": np.zeros((4, 4)), "tof": np.zeros((4, 4))}
    out = compare_results(a, b, {"q": 2.0}, {"q": 1.0})
    assert out["same_grid"] is False
    assert out["fields"] == {}
    assert out["summary"][0]["rel"] == pytest.approx(1.0)
