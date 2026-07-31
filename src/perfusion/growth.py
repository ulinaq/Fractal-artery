"""Vessel growth (service layer): space colonisation and CCO.

Two methods grow new branches onto an existing rooted
:class:`~perfusion.core.geometry.Network` so it fills a target tissue region:
:func:`grow_network` (space colonisation, a frontier drawn toward a point cloud)
and :func:`cco_grow` (constrained constructive optimisation, terminals added one
at a time by nearest-segment splitting). Both assign Murray-law radii and return
a solver-ready tree.

--- space colonisation ---

Grows new vessel branches onto an existing :class:`~perfusion.core.geometry.Network`
so it fills a target tissue region, using the *space-colonisation* algorithm
(Runions et al. 2007): the tree is drawn toward a cloud of **attraction points**
sampled in the tissue. Each iteration every attraction point pulls on the nearest
eligible tree node within an *influence radius*; each pulled node sprouts one new
segment of fixed *step* length in the average pull direction; attraction points
consumed within a *kill radius* of the tree are removed. Growth stops when the
points are exhausted, no node can grow, or a node cap is hit.

After the topology is grown, segment radii are assigned by **Murray's law**
(``r_parent^k = sum r_child^k``, leaves at ``leaf_radius``) so the result is a
physiologically tapered, solver-ready tree — feed it straight into
:func:`perfusion.core.pipeline.run_hybrid`.

This is a service built on the pure core: it depends only on NumPy/SciPy and
``core.geometry``. See ``docs/ROADMAP.md`` (Phase 3, vessel growth).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .core.geometry import Network, network_from_edges


def sample_points_in_mask(mask, domain_limit, n_points, *, seed=0, density=None):
    """Sample ``n_points`` random points inside the active cells of ``mask``.

    ``mask`` is an ``(nx, ny)`` boolean grid and ``domain_limit`` is
    ``(xmin, xmax, ymin, ymax)`` in physical coordinates (matching
    :class:`perfusion.core.darcy.Grid`). Points are drawn uniformly within the
    active cells, so they inherit the organ outline. Pass a ``density`` field
    (``(nx, ny)``, non-negative) to sample cells with probability proportional to it
    — data-constrained growth toward a target perfusion-density map.
    """
    mask = np.asarray(mask, dtype=bool)
    xmin, xmax, ymin, ymax = domain_limit
    nx, ny = mask.shape
    active = np.argwhere(mask)
    if active.size == 0:
        raise ValueError("mask has no active cells to sample")
    rng = np.random.default_rng(seed)
    if density is None:
        pick = rng.integers(0, len(active), size=n_points)
    else:
        # data-constrained: sample active cells with probability ∝ target density,
        # so attraction points (and hence vessels) concentrate where demand is high
        d = np.asarray(density, dtype=float)
        if d.shape != mask.shape:
            raise ValueError(f"density {d.shape} must match mask {mask.shape}")
        w = np.clip(d[active[:, 0], active[:, 1]], 0.0, None)
        total = w.sum()
        p = w / total if total > 0 else None      # fall back to uniform on a flat/empty map
        pick = rng.choice(len(active), size=n_points, p=p)
    ix, iy = active[pick, 0], active[pick, 1]
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (ix + rng.random(n_points)) * hx
    y = ymin + (iy + rng.random(n_points)) * hy
    return np.column_stack([x, y])


def _parent_nodes(base: Network) -> np.ndarray:
    """Parent-node index per node in ``base`` (-1 at the root)."""
    parent = np.full(base.n_nodes, -1, dtype=int)
    parent[base.seg_end] = base.seg_start
    return parent


def _murray_radii(parent, children, root, n_nodes, *, exponent, leaf_radius):
    """Radius of the segment feeding each node, by Murray's law (post-order)."""
    radius = np.zeros(n_nodes)
    postorder, stack = [], [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            postorder.append(node)
            continue
        stack.append((node, True))
        stack.extend((c, False) for c in children[node])
    for v in postorder:
        kids = children[v]
        if kids:
            radius[v] = sum(radius[c] ** exponent for c in kids) ** (1.0 / exponent)
        else:
            radius[v] = leaf_radius
    return radius


def grow_network(base: Network, attraction_points, *, seed_nodes=None,
                 step=None, influence_radius=None, kill_radius=None,
                 max_iter=400, max_nodes=5000, exponent=3.0,
                 leaf_radius=None) -> Network:
    """Grow ``base`` toward ``attraction_points`` by space colonisation.

    Parameters
    ----------
    base :
        The existing rooted network to grow from (e.g. an arterial trunk).
    attraction_points :
        ``(M, 2)`` array of target points in the tissue (see
        :func:`sample_points_in_mask`).
    seed_nodes :
        Node indices in ``base`` allowed to sprout. Defaults to the base's
        terminal nodes — growth extends the existing tips. Newly grown nodes are
        always eligible, so the tree advances into the point cloud.
    step, influence_radius, kill_radius :
        Space-colonisation length scales. Default to fractions of the combined
        bounding box of ``base`` and the points (``0.02``, ``0.25`` and
        ``2*step`` respectively) so the call works without tuning.
    max_iter, max_nodes :
        Safety caps on iterations and total nodes.
    exponent, leaf_radius :
        Murray's-law exponent and leaf radius for the final radius assignment.
        ``leaf_radius`` defaults to the thinnest segment in ``base``.

    Returns
    -------
    Network
        A new network (base topology + grown branches), rooted at ``base``'s
        root, with Murray-tapered radii.
    """
    pts = np.asarray(attraction_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("attraction_points must be an (M, 2) array")

    # Node arrays, seeded from the base network.
    pos = np.column_stack([base.node_x, base.node_y]).tolist()
    parent = _parent_nodes(base).tolist()
    if seed_nodes is None:
        seed_nodes = set(base.terminal_nodes.tolist())
    else:
        seed_nodes = {int(s) for s in seed_nodes}
    eligible = [i in seed_nodes for i in range(base.n_nodes)]

    # Length scales from the combined bounding box.
    all_x = np.concatenate([base.node_x, pts[:, 0]])
    all_y = np.concatenate([base.node_y, pts[:, 1]])
    extent = max(np.ptp(all_x), np.ptp(all_y), 1.0)
    step = float(step if step is not None else 0.02 * extent)
    influence_radius = float(influence_radius if influence_radius is not None else 0.25 * extent)
    kill_radius = float(kill_radius if kill_radius is not None else 2.0 * step)

    active = np.ones(len(pts), dtype=bool)
    for _ in range(max_iter):
        if not active.any() or len(pos) >= max_nodes:
            break
        node_arr = np.asarray(pos)

        # Consume attraction points already reached by any part of the tree.
        act_idx = np.flatnonzero(active)
        d_kill, _ = cKDTree(node_arr).query(pts[act_idx])
        active[act_idx[d_kill <= kill_radius]] = False
        if not active.any():
            break

        # Each remaining point pulls on its nearest *eligible* node.
        elig_idx = np.flatnonzero(eligible)
        if elig_idx.size == 0:
            break
        act_idx = np.flatnonzero(active)
        ap = pts[act_idx]
        d_inf, nearest = cKDTree(node_arr[elig_idx]).query(ap)
        within = d_inf <= influence_radius
        if not within.any():
            break

        # Accumulate the (normalised) pull direction on each influenced node.
        pull: dict[int, np.ndarray] = {}
        for k in np.flatnonzero(within):
            node = int(elig_idx[nearest[k]])
            v = ap[k] - node_arr[node]
            norm = np.hypot(v[0], v[1])
            if norm > 0:
                pull[node] = pull.get(node, np.zeros(2)) + v / norm

        grew = False
        for node, direction in pull.items():
            norm = np.hypot(direction[0], direction[1])
            if norm == 0:
                continue
            new_xy = node_arr[node] + step * direction / norm
            pos.append([float(new_xy[0]), float(new_xy[1])])
            parent.append(node)
            eligible.append(True)
            grew = True
        if not grew:
            break

    return _assemble(base, pos, parent, exponent=exponent, leaf_radius=leaf_radius)


def _assemble(base, pos, parent, *, exponent, leaf_radius):
    """Turn the grown (positions, parent) tree into a rooted :class:`Network`."""
    node_x = np.array([p[0] for p in pos])
    node_y = np.array([p[1] for p in pos])
    parent = np.asarray(parent, dtype=int)
    n_nodes = len(pos)

    children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for v in range(n_nodes):
        if parent[v] >= 0:
            children[parent[v]].append(v)

    if leaf_radius is None:
        leaf_radius = float(base.seg_radius.min())
    radius = _murray_radii(parent, children, base.root_node, n_nodes,
                           exponent=exponent, leaf_radius=leaf_radius)

    edges = []
    for v in range(n_nodes):
        p = parent[v]
        if p >= 0:
            length = float(np.hypot(node_x[v] - node_x[p], node_y[v] - node_y[p]))
            edges.append((int(p), int(v), max(length, 1e-9), float(radius[v])))
    return network_from_edges(node_x, node_y, edges, root=base.root_node)


def cco_grow(base: Network, candidates, *, n_terminals=200, d_min=None,
             exponent=3.0, leaf_radius=None, seed=0) -> Network:
    """Grow ``base`` by constrained constructive optimisation (CCO).

    Terminals are added **one at a time**: each new terminal point (drawn from
    ``candidates`` and kept only if it is at least ``d_min`` from every existing
    terminal — the *constraint*) is connected to the point on the existing tree
    nearest to it, splitting that segment with a new bifurcation (the greedy
    minimum-length *construction*). Radii follow Murray's law. This is the
    one-terminal-at-a-time counterpart to space-colonisation
    (:func:`grow_network`).

    Parameters
    ----------
    candidates : ``(M, 2)`` pool of possible terminal sites (see
        :func:`sample_points_in_mask`); consumed in a random order.
    n_terminals : how many terminals to add.
    d_min : minimum spacing between terminals; defaults to half the
        characteristic spacing ``sqrt(area / n_terminals)`` of the candidate box.
    """
    cand = np.asarray(candidates, dtype=float)
    if cand.ndim != 2 or cand.shape[1] != 2:
        raise ValueError("candidates must be an (M, 2) array")

    pos = np.column_stack([base.node_x, base.node_y]).tolist()
    parent = _parent_nodes(base).tolist()
    term = [pos[i] for i in base.terminal_nodes.tolist()] or [pos[base.root_node]]
    terminals = np.asarray(term, dtype=float)

    if d_min is None:
        area = max(np.ptp(cand[:, 0]) * np.ptp(cand[:, 1]), 1.0)
        d_min = 0.5 * np.sqrt(area / max(n_terminals, 1))

    rng = np.random.default_rng(seed)
    added = 0
    for idx in rng.permutation(len(cand)):
        if added >= n_terminals:
            break
        p = cand[idx]
        if np.min(np.hypot(terminals[:, 0] - p[0], terminals[:, 1] - p[1])) < d_min:
            continue                                   # too close to a terminal

        arr = np.asarray(pos)
        child = np.array([v for v in range(len(pos)) if parent[v] >= 0])
        if child.size == 0:                            # base is a lone node
            new_parent = base.root_node
        else:
            a, b = arr[[parent[v] for v in child]], arr[child]
            ab = b - a
            denom = np.einsum("ij,ij->i", ab, ab)
            s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1), 0, 1)
            m = a + s[:, None] * ab
            j = int(np.argmin(np.hypot(m[:, 0] - p[0], m[:, 1] - p[1])))
            v, pu, sj = int(child[j]), int(parent[child[j]]), float(s[j])
            if sj <= 1e-6:
                new_parent = pu
            elif sj >= 1 - 1e-6:
                new_parent = v
            else:                                       # split pu->v at the foot m_j
                new_parent = len(pos)
                pos.append([float(m[j, 0]), float(m[j, 1])])
                parent.append(pu)
                parent[v] = new_parent
        pos.append([float(p[0]), float(p[1])])
        parent.append(new_parent)
        terminals = np.vstack([terminals, p])
        added += 1

    return _assemble(base, pos, parent, exponent=exponent, leaf_radius=leaf_radius)


def cco_grow_in_mask(base: Network, mask, domain_limit, *, n_terminals=200,
                     n_pool=None, seed=0, density=None, **kwargs) -> Network:
    """Convenience: sample a candidate pool in ``mask`` then :func:`cco_grow`.

    ``density`` (an ``(nx, ny)`` target map) biases the candidate pool toward
    high-demand tissue (data-constrained growth)."""
    pool = sample_points_in_mask(mask, domain_limit, n_pool or 6 * n_terminals,
                                 seed=seed, density=density)
    return cco_grow(base, pool, n_terminals=n_terminals, seed=seed, **kwargs)


def grow_in_mask(base: Network, mask, domain_limit, *, n_points=400, seed=0,
                 density=None, **kwargs) -> Network:
    """Convenience: sample attraction points in ``mask`` then :func:`grow_network`.

    Pass ``density`` (an ``(nx, ny)`` target map) to bias the attraction points toward
    high-demand tissue (data-constrained growth). Any space-colonisation keyword
    (``step``, ``influence_radius``, …) is forwarded to :func:`grow_network`.
    """
    pts = sample_points_in_mask(mask, domain_limit, n_points, seed=seed, density=density)
    return grow_network(base, pts, **kwargs)
