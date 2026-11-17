"""Red-blood-cell phase separation at bifurcations (Pries plasma skimming).

At a diverging bifurcation red cells do **not** split in proportion to the blood
flow: the lower-flow daughter is skimmed of cells (can even receive pure plasma).
This makes the discharge hematocrit H_D vary segment-to-segment, and since blood
viscosity depends on H_D (Fåhræus-Lindqvist), it feeds back on the flow.

`phase_separation_fraction` is the empirical Pries, Secomb & Gaehtgens (1990) law;
`segment_hematocrit` propagates H_D from the arterial root through the tree given
the per-segment flows. Diameters are in micrometres. Pure NumPy — lives in
``core`` and couples into ``pipeline.run_hybrid`` behind a flag.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def phase_separation_fraction(fq_blood: float, hd_parent: float,
                              d_parent: float, d_alpha: float, d_beta: float) -> float:
    """Fraction of the parent's RBC flux entering daughter α (Pries 1990).

    ``fq_blood`` is the fraction of the parent's *blood* flow going to α. Returns
    the fraction of its *red-cell* flow going to α. Symmetric daughters at a 50/50
    split give 0.5; a low-flow daughter (below ``X0 = 0.4/D_parent``) is skimmed to
    0; the larger daughter draws proportionally more cells.
    """
    if d_parent <= 0:
        return fq_blood
    x0 = 0.4 / d_parent
    if fq_blood <= x0:
        return 0.0
    if fq_blood >= 1.0 - x0:
        return 1.0
    a = -6.96 * np.log(d_alpha / d_beta) / d_parent
    b = 1.0 + 6.98 * (1.0 - hd_parent) / d_parent
    z = (fq_blood - x0) / (1.0 - 2.0 * x0)
    logit = a + b * np.log(z / (1.0 - z))
    return float(1.0 / (1.0 + np.exp(-logit)))


def segment_hematocrit(net, flow, inlet_hd: float = 0.45,
                       radius_to_um: float = 1.0) -> np.ndarray:
    """Per-segment discharge hematocrit from RBC phase separation.

    Propagates H_D from ``net.root_node`` downstream (segments oriented
    start→end), splitting red cells at binary diverging bifurcations by
    :func:`phase_separation_fraction` and mixing (RBC-flux-conserving) elsewhere.
    ``flow`` is the per-segment blood-flow magnitude. RBC flux is conserved at
    every node, so arterial inflow hematocrit is preserved on average.
    """
    flow = np.abs(np.asarray(flow, dtype=float))
    m = int(net.n_segments)
    hd = np.full(m, float(inlet_hd))
    if m == 0:
        return hd
    d = 2.0 * np.asarray(net.seg_radius, dtype=float) * radius_to_um

    children: dict[int, list[int]] = defaultdict(list)
    for s in range(m):
        children[int(net.seg_start[s])].append(s)
    parent_seg = net.incoming_segment
    root = int(net.root_node)

    for s in children[root]:               # trunk carries the systemic hematocrit
        hd[s] = inlet_hd

    order, seen, dq = [], {root}, deque([root])
    while dq:                              # BFS so a node's parent is set before it
        node = dq.popleft()
        order.append(node)
        for s in children[node]:
            c = int(net.seg_end[s])
            if c not in seen:
                seen.add(c)
                dq.append(c)

    for node in order:
        ps = int(parent_seg[node])
        ch = children[node]
        if ps < 0 or not ch:              # root feed already set; leaves have no children
            continue
        qp, rp = flow[ps], hd[ps] * flow[ps]
        if qp <= 0:
            for s in ch:
                hd[s] = 0.0
            continue
        if len(ch) == 2:                  # binary diverging bifurcation -> Pries skimming
            c1, c2 = ch
            q1, q2 = flow[c1], flow[c2]
            if q1 + q2 <= 0:
                hd[c1] = hd[c2] = 0.0
                continue
            fe1 = phase_separation_fraction(q1 / (q1 + q2), hd[ps], d[ps], d[c1], d[c2])
            hd[c1] = min(fe1 * rp / q1, 1.0) if q1 > 0 else 0.0
            hd[c2] = min((1.0 - fe1) * rp / q2, 1.0) if q2 > 0 else 0.0
        else:                             # pass-through / multifurcation: well-mixed
            hbar = min(rp / qp, 1.0)
            for s in ch:
                hd[s] = hbar
    return hd
