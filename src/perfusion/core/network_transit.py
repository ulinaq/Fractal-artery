"""Network-transit kinetics for the direct-connection model (Paper C).

The direct model has no Darcy field, so a bolus cannot be advected on a grid.
Instead we compute the *transit time* of every vascular element — arterial
segment, capillary connection, venous segment — as ``τ = volume / |flow|``, and
accumulate it along the flow direction (arterial root → terminals → connections →
venous root) to get each element's **arrival time**. A contrast bolus injected at
the arterial inlet then reaches each element delayed by its arrival time, giving a
network animation that mirrors the field tracer without a grid.

Flow direction is taken straight from the solved node pressures (high → low), so
the construction is a DAG and is robust to the loops that venous graphs legitimately
contain — no reliance on segment orientation conventions.
"""

from __future__ import annotations

import numpy as np

from .direct import Connection, DirectSolution
from .geometry import Network

_EPS = 1e-12


def _segment_volumes(net: Network) -> np.ndarray:
    """Luminal volume ``π r² L`` per segment."""
    return np.pi * net.seg_radius**2 * net.seg_length


def arrival_times(art: Network, ven: Network, connections: list[Connection],
                  sol: DirectSolution) -> tuple[np.ndarray, np.ndarray]:
    """Bolus arrival time at every arterial and venous node.

    Builds the flow-directed graph (each edge points from the higher-pressure to
    the lower-pressure end, weight = element transit time ``V/|q|``) and relaxes
    arrivals in descending-pressure order from the arterial inlet(s) at ``t=0``.
    Because pressure decreases monotonically along flow, this order finalises every
    node before its successors — a topological sweep, no heap needed. Returns
    ``(T_art, T_ven)``; unreached / stagnant nodes stay ``inf``.
    """
    na, nv = art.n_nodes, ven.n_nodes
    pressure = np.concatenate([sol.p_a, sol.p_v])
    # Negligible-flow elements carry no bolus: below this they are non-conducting,
    # so stagnant branches stay dark rather than acquiring absurd V/|q| transit times.
    q_all = np.concatenate([np.abs(sol.q_a), np.abs(sol.q_v), np.abs(sol.q_c)])
    q_tol = 1e-9 * (float(q_all.max()) if q_all.size else 0.0)
    edges: list[tuple[int, int, float]] = []       # (upstream, downstream, weight)

    def tree_edges(net: Network, q: np.ndarray, p: np.ndarray, offset: int) -> None:
        vol = _segment_volumes(net)
        for s in range(net.n_segments):
            if abs(q[s]) <= q_tol:
                continue
            u, v = int(net.seg_start[s]), int(net.seg_end[s])
            if p[v] > p[u]:                          # orient high → low pressure
                u, v = v, u
            edges.append((offset + u, offset + v, float(vol[s] / abs(q[s]))))

    tree_edges(art, sol.q_a, sol.p_a, 0)
    tree_edges(ven, sol.q_v, sol.p_v, na)
    for c, qc in zip(connections, sol.q_c, strict=True):
        if abs(float(qc)) <= q_tol:
            continue
        vol = np.pi * c.radius**2 * c.length
        edges.append((c.arterial_terminal, na + c.venous_terminal, vol / abs(float(qc))))

    n = na + nv
    out: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        out[u].append((v, w))

    arrival = np.full(n, np.inf)
    for r in art.inlet_nodes:
        arrival[int(r)] = 0.0
    for u in np.argsort(pressure)[::-1]:            # descending pressure = flow order
        au = arrival[u]
        if not np.isfinite(au):
            continue
        for v, w in out[u]:
            if au + w < arrival[v]:
                arrival[v] = au + w
    return arrival[:na], arrival[na:]


def transit_animation(art: Network, ven: Network, connections: list[Connection],
                      sol: DirectSolution, n_frames: int = 48, disp_frac: float = 0.05,
                      shape: float = 3.0) -> dict:
    """Per-element bolus animation for the direct model.

    Each element lights up with a gamma-variate bolus ``(x/B)^A e^{-x/B}`` in
    ``x = t − T`` (its arrival time ``T``), so the wave sweeps arteries → capillary
    connections → veins. Returns a dict of arrays ready to persist / serialise:

    - ``lines`` ``(n_elem, 4)`` — ``[x0, y0, x1, y1]`` per element,
    - ``kind`` ``(n_elem,)`` — 0 arterial, 1 venous, 2 connection,
    - ``arrivals`` ``(n_elem,)`` — element arrival time (``inf`` → never reached),
    - ``frames`` ``(n_frames, n_elem)`` — normalised concentration (peak 1),
    - ``times`` ``(n_frames,)`` and ``vmax`` (== 1.0).
    """
    t_art, t_ven = arrival_times(art, ven, connections, sol)

    def seg_lines(net: Network) -> np.ndarray:
        return np.column_stack([
            net.node_x[net.seg_start], net.node_y[net.seg_start],
            net.node_x[net.seg_end], net.node_y[net.seg_end],
        ]).astype(float)

    a_lines, v_lines = seg_lines(art), seg_lines(ven)
    # an element's arrival is when the bolus enters it — its upstream (earlier) end
    a_arr = np.minimum(t_art[art.seg_start], t_art[art.seg_end])
    v_arr = np.minimum(t_ven[ven.seg_start], t_ven[ven.seg_end])
    if connections:
        c_lines = np.array([[art.node_x[c.arterial_terminal], art.node_y[c.arterial_terminal],
                             ven.node_x[c.venous_terminal], ven.node_y[c.venous_terminal]]
                            for c in connections], dtype=float)
        c_arr = np.array([t_art[c.arterial_terminal] for c in connections], dtype=float)
    else:
        c_lines, c_arr = np.zeros((0, 4)), np.zeros(0)

    lines = np.vstack([a_lines, v_lines, c_lines])
    kind = np.concatenate([np.zeros(len(a_lines)), np.ones(len(v_lines)),
                           np.full(len(c_lines), 2)]).astype(np.int8)
    arrivals = np.concatenate([a_arr, v_arr, c_arr])

    finite = arrivals[np.isfinite(arrivals)]
    t_max = float(finite.max()) if finite.size else 1.0
    disp = max(disp_frac * t_max, _EPS)                      # bolus dispersion width B
    duration = 1.25 * t_max + shape * disp + 6.0 * disp
    times = np.linspace(0.0, duration, int(n_frames))

    x = np.clip(times[:, None] - arrivals[None, :], 0.0, None)   # (n_frames, n_elem)
    frames = x**shape * np.exp(-x / disp)
    peak = (shape * disp) ** shape * np.exp(-shape)              # analytic gamma peak
    frames = (frames / peak).astype(np.float32)
    frames[:, ~np.isfinite(arrivals)] = 0.0                     # unreached → dark

    return {"lines": lines, "kind": kind, "arrivals": arrivals,
            "frames": frames, "times": times, "vmax": 1.0}
