"""Comparing two solved perfusion results (run A/B — e.g. before/after a lesion,
a pressure change or a regrown tree).

Pure NumPy: works on the stored result fields (``pressure`` / ``flux`` / ``tof`` and
the ``mask``), 2-D or 3-D alike. A difference is always ``current − baseline``.
"""

from __future__ import annotations

import numpy as np

FIELD_NAMES = ("pressure", "flux", "tof")


def field_difference(current, baseline, mask_current=None, mask_baseline=None):
    """``current − baseline`` on the cells active (and finite) in both; NaN elsewhere.

    The two fields must share a grid — compare runs of the same project geometry."""
    a, b = np.asarray(current, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"the grids differ ({b.shape} vs {a.shape}); a difference map "
                         "needs the same grid in both runs")
    ok = np.isfinite(a) & np.isfinite(b)
    for m in (mask_current, mask_baseline):
        if m is not None:
            ok &= np.asarray(m, dtype=bool)
    return np.where(ok, a - b, np.nan)


def field_stats(current, baseline, mask_current=None, mask_baseline=None) -> dict:
    """Summary of one field's change over the shared active cells: the mean in each run,
    the mean and max absolute difference, and the mean relative change."""
    d = field_difference(current, baseline, mask_current, mask_baseline)
    ok = np.isfinite(d)
    if not ok.any():
        return {"mean_baseline": None, "mean_current": None, "mean_abs_diff": None,
                "max_abs_diff": None, "rel_change": None, "cells": 0}
    a, b = np.asarray(current, dtype=float)[ok], np.asarray(baseline, dtype=float)[ok]
    mb, mc = float(b.mean()), float(a.mean())
    return {"mean_baseline": mb, "mean_current": mc,
            "mean_abs_diff": float(np.abs(d[ok]).mean()),
            "max_abs_diff": float(np.abs(d[ok]).max()),
            "rel_change": (mc - mb) / abs(mb) if mb != 0 else None,
            "cells": int(ok.sum())}


def summary_delta(current: dict, baseline: dict) -> list[dict]:
    """Per-quantity change between two result summaries (the numeric keys both share):
    ``{key, baseline, current, diff, rel}`` with ``rel`` = diff / |baseline|."""
    rows = []
    for key in baseline:
        b, a = baseline.get(key), current.get(key)
        if isinstance(b, bool) or isinstance(a, bool):
            continue
        if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        diff = float(a) - float(b)
        rows.append({"key": key, "baseline": float(b), "current": float(a), "diff": diff,
                     "rel": diff / abs(b) if b != 0 else None})
    return rows


def compare_results(current: dict, baseline: dict, summary_current: dict | None = None,
                    summary_baseline: dict | None = None) -> dict:
    """Full A/B comparison of two result blobs (dicts of arrays with ``mask`` and the
    fields): per-field change statistics and, given the summaries, per-quantity
    deltas. Grids that differ (e.g. after a resolution change) still compare by
    summary; ``same_grid`` says whether the field statistics are available."""
    same = np.shape(current["pressure"]) == np.shape(baseline["pressure"])
    fields = ({name: field_stats(current[name], baseline[name], current.get("mask"),
                                 baseline.get("mask"))
               for name in FIELD_NAMES} if same else {})
    out = {"same_grid": bool(same), "fields": fields}
    if summary_current is not None and summary_baseline is not None:
        out["summary"] = summary_delta(summary_current, summary_baseline)
    return out
