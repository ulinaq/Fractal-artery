"""Anastomoses — vessel loops (collaterals) added to a tree.

Real vasculature is not a pure tree: arcades (mesentery), the circle of Willis and
coronary / cerebral collaterals join branches so blood can reach a territory by a
second route when one is blocked. :func:`add_anastomoses` adds such **side-to-side**
links: two nearby segments on *different* branches — far apart along the tree,
``detour = tree path / straight distance`` large — are split at their midpoints and the
midpoints joined. Existing nodes keep their indices and every terminal stays a
terminal, so a territory (the terminals downstream of a segment) is well defined
before and after; :func:`occlude` and :func:`downstream_terminals` measure how much
flow loops rescue when a branch is blocked.

The network solver is incidence-based and already handles loops; nothing else
changes. Works on 2-D and 3-D networks (positions via ``Network.positions``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .geometry import Network, NodeRole


def n_loops(net: Network) -> int:
    """Independent loops (cyclomatic number) ``E − V + C`` — 0 for a forest."""
    if net.n_segments == 0:
        return 0
    g = coo_matrix((np.ones(net.n_segments), (net.seg_start, net.seg_end)),
                   shape=(net.n_nodes, net.n_nodes))
    used = net.node_degree > 0
    _, labels = connected_components(g, directed=False)
    comps = len(np.unique(labels[used]))
    return int(net.n_segments - int(used.sum()) + comps)


def _graph(net: Network):
    return coo_matrix((net.seg_length, (net.seg_start, net.seg_end)),
                      shape=(net.n_nodes, net.n_nodes)).tocsr()


def _split(nodes: dict, segs: dict, s: int, mid: np.ndarray) -> int:
    """Split segment ``s`` at ``mid``: ``s`` becomes start→mid, a new segment mid→end.
    Returns the new node's index."""
    m = len(nodes["x"])
    nodes["x"].append(mid[0])
    nodes["y"].append(mid[1])
    if nodes["z"] is not None:
        nodes["z"].append(mid[2])
    end = segs["end"][s]
    half = segs["length"][s] / 2.0
    segs["end"][s] = m
    segs["length"][s] = half
    for key, val in (("start", m), ("end", end), ("length", half),
                     ("radius", segs["radius"][s]), ("angle", segs["angle"][s])):
        segs[key].append(val)
    return m


def add_anastomoses(net: Network, n_links: int = 4, *, max_distance: float | None = None,
                    min_detour: float = 3.0, min_hops: int = 3, radius_frac: float = 0.7,
                    skip_trunk: int = 1) -> tuple[Network, list[dict]]:
    """Add up to ``n_links`` side-to-side anastomoses (loops) to ``net``.

    Candidates are segment midpoints — except the ``skip_trunk`` segment generations
    nearest an inlet/outlet (no shunt across the trunk). A pair qualifies when the two
    midpoints are within ``max_distance`` (default 3 × the median segment length) and
    the tree path between them is at least ``min_detour`` × their straight distance and
    at least ``min_hops`` junctions long (so siblings off one junction never link —
    the loop must join different branches). The closest qualifying pairs are linked first;
    path lengths are recomputed after each link so loops don't duplicate a route, and
    each segment is split at most once. The link radius is ``radius_frac`` × the
    thinner of the two segments; its length is the straight distance.

    Returns ``(network, links)`` — ``links`` lists ``{a, b, length, radius,
    segments}`` (new midpoint nodes and the original segments they split). Node
    indices of ``net`` are preserved (new nodes are appended); terminals are unchanged.
    """
    if n_links < 0:
        raise ValueError("n_links must be >= 0")
    if min_hops < 1:
        raise ValueError("min_hops must be >= 1")
    if min_detour <= 1.0:
        raise ValueError("min_detour must be > 1 (a loop needs a detour to bypass)")
    if not 0 < radius_frac <= 1.0:
        raise ValueError("radius_frac must be in (0, 1]")
    if net.n_segments < 2 or n_links == 0:
        return net, []
    pos = net.positions()
    mids = 0.5 * (pos[net.seg_start] + pos[net.seg_end])
    if max_distance is None:
        max_distance = 3.0 * float(np.median(net.seg_length))

    # generations from the inlets (in segments): skip the trunk
    hops = dijkstra(coo_matrix((np.ones(net.n_segments), (net.seg_start, net.seg_end)),
                               shape=(net.n_nodes, net.n_nodes)).tocsr(),
                    directed=False, indices=np.asarray(net.inlet_nodes), min_only=True)
    seg_gen = np.minimum(hops[net.seg_start], hops[net.seg_end])
    eligible = np.flatnonzero(np.isfinite(seg_gen) & (seg_gen >= skip_trunk))
    if eligible.size < 2:
        return net, []
    pairs = np.array(sorted(cKDTree(mids[eligible]).query_pairs(max_distance)), dtype=int)
    if pairs.size == 0:
        return net, []
    pairs = eligible[pairs]                                     # segment index pairs
    gap = np.linalg.norm(mids[pairs[:, 0]] - mids[pairs[:, 1]], axis=1)
    order = np.argsort(gap, kind="stable")
    pairs, gap = pairs[order], gap[order]

    nodes = {"x": list(net.node_x), "y": list(net.node_y),
             "z": None if net.node_z is None else list(net.node_z)}
    segs = {"start": list(net.seg_start), "end": list(net.seg_end),
            "length": list(net.seg_length.astype(float)),
            "radius": list(net.seg_radius.astype(float)),
            "angle": list(net.seg_angle.astype(float))}
    split = set()
    links: list[dict] = []
    work = net
    for (s1, s2), d in zip(pairs, gap, strict=True):
        if len(links) >= n_links:
            break
        if s1 in split or s2 in split or d <= 0:
            continue
        # tree path between the two midpoints, on the current network
        ends1 = (int(work.seg_start[s1]), int(work.seg_end[s1]))
        ends2 = (int(work.seg_start[s2]), int(work.seg_end[s2]))
        dist = dijkstra(_graph(work), directed=False, indices=list(ends1))
        path = min(dist[i, j] for i in range(2) for j in ends2) \
            + 0.5 * (work.seg_length[s1] + work.seg_length[s2])
        if not np.isfinite(path) or path < min_detour * d:
            continue
        hop = dijkstra(_graph(replace(work, seg_length=np.ones(work.n_segments))),
                       directed=False, indices=list(ends1))
        if min(hop[i, j] for i in range(2) for j in ends2) + 1 < min_hops:
            continue
        a = _split(nodes, segs, s1, mids[s1])
        b = _split(nodes, segs, s2, mids[s2])
        r = radius_frac * min(segs["radius"][s1], segs["radius"][s2])
        delta = mids[s2] - mids[s1]
        for key, val in (("start", a), ("end", b), ("length", float(d)), ("radius", r),
                         ("angle", float(np.arctan2(delta[1], delta[0])))):
            segs[key].append(val)
        split.update((s1, s2))
        links.append({"a": a, "b": b, "length": float(d), "radius": float(r),
                      "segments": [int(s1), int(s2)]})
        work = _assemble(net, nodes, segs)
    return (work if links else net), links


def _assemble(template: Network, nodes: dict, segs: dict) -> Network:
    ss, se = np.asarray(segs["start"], int), np.asarray(segs["end"], int)
    n = len(nodes["x"])
    deg = np.bincount(np.concatenate([ss, se]), minlength=n)
    role = np.where(deg >= 2, int(NodeRole.BIFURCATION), int(NodeRole.TERMINAL))
    role[template.inlet_nodes] = int(NodeRole.ROOT)
    return Network(
        node_x=np.asarray(nodes["x"], float), node_y=np.asarray(nodes["y"], float),
        node_role=role, seg_start=ss, seg_end=se,
        seg_length=np.asarray(segs["length"], float),
        seg_radius=np.asarray(segs["radius"], float),
        seg_angle=np.asarray(segs["angle"], float), root_node=template.root_node,
        inlet_nodes=np.asarray(template.inlet_nodes, int),
        node_z=None if nodes["z"] is None else np.asarray(nodes["z"], float))


def downstream_terminals(tree: Network, seg: int) -> np.ndarray:
    """Terminals fed through segment ``seg`` of a **tree** (its territory): every
    terminal whose path to the inlets crosses ``seg``. Orientation-free — found by
    removing ``seg`` and taking the side cut off from all inlets."""
    keep = np.ones(tree.n_segments, bool)
    keep[seg] = False
    g = coo_matrix((np.ones(int(keep.sum())), (tree.seg_start[keep], tree.seg_end[keep])),
                   shape=(tree.n_nodes, tree.n_nodes))
    _, labels = connected_components(g, directed=False)
    fed = np.isin(labels, labels[np.asarray(tree.inlet_nodes)])
    term = tree.terminal_nodes
    return term[~fed[term]]


def occlude(net: Network, segments, factor: float = 1e-3) -> Network:
    """A copy with ``segments`` narrowed to ``factor`` × their radius — an embolus /
    stenosis (Poiseuille resistance grows as ``factor⁻⁴``)."""
    r = net.seg_radius.astype(float).copy()
    r[np.atleast_1d(np.asarray(segments, int))] *= float(factor)
    return replace(net, seg_radius=r)
