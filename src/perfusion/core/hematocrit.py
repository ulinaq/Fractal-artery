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

    ``flow`` is the per-segment blood flow, signed ``seg_start → seg_end`` (a
    magnitude works for a tree whose segments all point downstream). Red cells are
    followed *with the flow*: nodes are visited in flow order (upstream first), each
    node pools the RBC flux of **every** segment feeding it — so a loop node (two
    parents, e.g. an anastomosis) mixes both — and splits it among the segments
    leaving it: by :func:`phase_separation_fraction` at a binary diverging
    bifurcation (the parent diameter is the widest feeder), well-mixed otherwise.
    Segments leaving a node with no inflow (the inlet) carry ``inlet_hd``. RBC flux
    is conserved at every node.
    """
    q = np.asarray(flow, dtype=float)
    m = int(net.n_segments)
    hd = np.full(m, float(inlet_hd))
    if m == 0:
        return hd
    d = 2.0 * np.asarray(net.seg_radius, dtype=float) * radius_to_um
    ss, se = np.asarray(net.seg_start, int), np.asarray(net.seg_end, int)
    up = np.where(q >= 0, ss, se)            # flow runs up → down
    down = np.where(q >= 0, se, ss)
    mag = np.abs(q)

    feeds: dict[int, list[int]] = defaultdict(list)
    drains: dict[int, list[int]] = defaultdict(list)
    for s in range(m):
        feeds[int(down[s])].append(s)
        drains[int(up[s])].append(s)

    # topological order of the flow graph (Kahn): steady potential flow has no
    # directed cycles; any node left over (stagnant ties) is appended in index order
    indeg = {n: len(feeds[n]) for n in range(net.n_nodes)}
    dq = deque(n for n in range(net.n_nodes) if indeg[n] == 0)
    order, done = [], set()
    while dq:
        node = dq.popleft()
        order.append(node)
        done.add(node)
        for s in drains[node]:
            c = int(down[s])
            indeg[c] -= 1
            if indeg[c] == 0:
                dq.append(c)
    order += [n for n in range(net.n_nodes) if n not in done]

    for node in order:
        out = drains[node]
        if not out:
            continue
        inn = feeds[node]
        if not inn:                          # an inlet: the systemic hematocrit
            for s in out:
                hd[s] = inlet_hd
            continue
        q_in = float(mag[inn].sum())
        rbc_in = float((hd[inn] * mag[inn]).sum())
        if q_in <= 0:
            for s in out:
                hd[s] = 0.0
            continue
        h_in = min(rbc_in / q_in, 1.0)
        if len(out) == 2:                    # binary diverging bifurcation -> skimming
            c1, c2 = out
            q1, q2 = mag[c1], mag[c2]
            if q1 + q2 <= 0:
                hd[c1] = hd[c2] = 0.0
                continue
            d_par = float(d[inn].max())
            fe1 = phase_separation_fraction(q1 / (q1 + q2), h_in, d_par, d[c1], d[c2])
            rbc_out = h_in * (q1 + q2)        # what leaves (= what arrived, by mass balance)
            hd[c1] = min(fe1 * rbc_out / q1, 1.0) if q1 > 0 else 0.0
            hd[c2] = min((1.0 - fe1) * rbc_out / q2, 1.0) if q2 > 0 else 0.0
        else:                                # pass-through / multifurcation: well-mixed
            for s in out:
                hd[s] = h_in
    return hd
