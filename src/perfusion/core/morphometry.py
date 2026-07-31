"""Horton–Strahler morphometry of a vascular tree.

Assigns a Strahler order to every segment (leaves = order 1; where two or more
same-order-``ω`` branches meet the parent becomes ``ω+1``, otherwise it keeps the
maximum child order) and summarises the classic morphometric laws — the bifurcation
ratio ``R_B`` (how fast segment counts grow toward the root), the length ratio ``R_L``
and the diameter ratio ``R_D``. Used to characterise and validate grown / data-
constrained trees against measured vasculature.

Orders are computed on the segment orientation ``seg_start → seg_end`` (parent →
child, matching :attr:`Network.incoming_segment`); for the rare looped venous graph
this uses the rooted spanning orientation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import Network


def strahler_orders(net: Network) -> np.ndarray:
    """Per-segment Horton–Strahler order (``(n_segments,)`` int array)."""
    n = net.n_segments
    if n == 0:
        return np.zeros(0, dtype=int)
    start, end = np.asarray(net.seg_start), np.asarray(net.seg_end)
    children: list[list[int]] = [[] for _ in range(n)]
    by_start: dict[int, list[int]] = {}
    for s in range(n):
        by_start.setdefault(int(start[s]), []).append(s)
    for s in range(n):                          # child segments continue from this segment's end
        children[s] = by_start.get(int(end[s]), [])

    # process leaves-first by descending downstream-node depth (BFS from the root nodes)
    depth = np.full(net.n_nodes, -1, dtype=int)
    roots = deque(int(r) for r in np.flatnonzero(net.incoming_segment < 0))
    for r in roots:
        depth[r] = 0
    q = deque(roots)
    while q:
        node = q.popleft()
        for s in by_start.get(node, []):
            child = int(end[s])
            if depth[child] < 0:
                depth[child] = depth[node] + 1
                q.append(child)
    seg_depth = np.where(depth[end] >= 0, depth[end], 0)
    order = np.zeros(n, dtype=int)
    for s in sorted(range(n), key=lambda i: seg_depth[i], reverse=True):
        kids = [order[c] for c in children[s] if order[c] > 0]
        if not kids:
            order[s] = 1
        else:
            top = max(kids)
            order[s] = top + 1 if kids.count(top) >= 2 else top
    return order


def _ratio(orders: np.ndarray, values: np.ndarray, kind: str) -> tuple[float, dict]:
    """Fit a Horton geometric law across orders; return ``(ratio, per-order means)``.

    ``kind='count'`` fits ``N_ω = R^{Ω-ω}`` (``R = e^{-slope}`` of ``log N`` vs ω);
    ``kind='mean'`` fits ``X_ω = X_1 R^{ω-1}`` (``R = e^{+slope}``).
    """
    ws = np.unique(orders)
    if kind == "count":
        y = np.array([np.sum(orders == w) for w in ws], dtype=float)
        per = {int(w): int(c) for w, c in zip(ws, y, strict=True)}
    else:
        y = np.array([values[orders == w].mean() for w in ws], dtype=float)
        per = {int(w): float(v) for w, v in zip(ws, y, strict=True)}
    if len(ws) < 2 or np.any(y <= 0):
        return float("nan"), per
    slope = np.polyfit(ws.astype(float), np.log(y), 1)[0]
    ratio = float(np.exp(-slope if kind == "count" else slope))
    return ratio, per


def strahler_summary(net: Network) -> dict:
    """Strahler orders + Horton morphometric ratios for ``net``.

    Returns the max order, per-order segment counts / mean length / mean radius, and
    the bifurcation (``R_B``), length (``R_L``) and diameter (``R_D``) ratios.
    """
    order = strahler_orders(net)
    valid = order > 0
    orders = order[valid]
    if orders.size == 0:
        return {"max_order": 0, "counts": {}, "mean_length": {}, "mean_radius": {},
                "bifurcation_ratio": float("nan"), "length_ratio": float("nan"),
                "diameter_ratio": float("nan")}
    length = np.asarray(net.seg_length)[valid]
    radius = np.asarray(net.seg_radius)[valid]
    r_b, counts = _ratio(orders, orders, "count")
    r_l, mean_len = _ratio(orders, length, "mean")
    r_d, mean_rad = _ratio(orders, 2.0 * radius, "mean")   # diameters
    return {
        "max_order": int(orders.max()),
        "counts": counts, "mean_length": mean_len, "mean_radius": mean_rad,
        "bifurcation_ratio": r_b, "length_ratio": r_l, "diameter_ratio": r_d,
    }
