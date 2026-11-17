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

    ``mask`` is an ``(nx, ny)`` boolean grid with ``domain_limit`` =
    ``(xmin, xmax, ymin, ymax)`` (matching :class:`perfusion.core.darcy.Grid`), or
    an ``(nx, ny, nz)`` grid with ``(xmin, xmax, ymin, ymax, zmin, zmax)`` for 3-D
    growth. Points are drawn uniformly within the active cells, so they inherit
    the organ outline. Pass a ``density`` field (same shape as ``mask``,
    non-negative) to sample cells with probability proportional to it —
    data-constrained growth toward a target perfusion-density map.
    """
    mask = np.asarray(mask, dtype=bool)
    dim = mask.ndim
    if len(domain_limit) != 2 * dim:
        raise ValueError(f"a {dim}-D mask needs a {2 * dim}-value domain_limit")
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
        w = np.clip(d[tuple(active.T)], 0.0, None)
        total = w.sum()
        p = w / total if total > 0 else None      # fall back to uniform on a flat/empty map
        pick = rng.choice(len(active), size=n_points, p=p)
    lo = np.asarray(domain_limit[0::2], dtype=float)
    h = (np.asarray(domain_limit[1::2], dtype=float) - lo) / np.asarray(mask.shape)
    return lo + (active[pick] + rng.random((n_points, dim))) * h


def _positions(net: Network) -> np.ndarray:
    """Node coordinates as an ``(n, 2)`` or ``(n, 3)`` array."""
    cols = [net.node_x, net.node_y] + ([net.node_z] if net.is_3d else [])
    return np.column_stack(cols)


def _check_points(net: Network, pts, name):
    pts = np.asarray(pts, dtype=float)
    dim = 3 if net.is_3d else 2
    if pts.ndim != 2 or pts.shape[1] != dim:
        raise ValueError(f"{name} must be an (M, {dim}) array for a {dim}-D tree")
    return pts


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
    pts = _check_points(base, attraction_points, "attraction_points")

    # Node arrays, seeded from the base network (2-D or 3-D).
    pos = _positions(base).tolist()
    parent = _parent_nodes(base).tolist()
    if seed_nodes is None:
        seed_nodes = set(base.terminal_nodes.tolist())
    else:
        seed_nodes = {int(s) for s in seed_nodes}
    eligible = [i in seed_nodes for i in range(base.n_nodes)]

    # Length scales from the combined bounding box.
    extent = max(float(np.ptp(np.vstack([_positions(base), pts]), axis=0).max()), 1.0)
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
            norm = np.linalg.norm(v)
            if norm > 0:
                pull[node] = pull.get(node, 0.0) + v / norm

        grew = False
        for node, direction in pull.items():
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            new_xy = node_arr[node] + step * direction / norm
            pos.append([float(c) for c in new_xy])
            parent.append(node)
            eligible.append(True)
            grew = True
        if not grew:
            break

    return _assemble(base, pos, parent, exponent=exponent, leaf_radius=leaf_radius)


def _assemble(base, pos, parent, *, exponent, leaf_radius):
    """Turn the grown (positions, parent) tree into a rooted :class:`Network`."""
    P = np.asarray(pos, dtype=float)
    node_x, node_y = P[:, 0], P[:, 1]
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

    if P.shape[1] == 3:
        return _assemble_3d(base, P, parent, radius)
    edges = []
    for v in range(n_nodes):
        p = parent[v]
        if p >= 0:
            length = float(np.hypot(node_x[v] - node_x[p], node_y[v] - node_y[p]))
            edges.append((int(p), int(v), max(length, 1e-9), float(radius[v])))
    return network_from_edges(node_x, node_y, edges, root=base.root_node)


def _assemble_3d(base, P, parent, radius):
    """3-D twin of the 2-D assembly: the grown tree is already parent→child
    oriented from ``base``'s root, so build the :class:`Network` directly."""
    from .core.geometry import NodeRole

    child = np.flatnonzero(parent >= 0)
    ss, se = parent[child], child
    deg = np.bincount(np.concatenate([ss, se]), minlength=len(P))
    role = np.where(deg >= 2, int(NodeRole.BIFURCATION), int(NodeRole.TERMINAL))
    role[base.root_node] = int(NodeRole.ROOT)
    d = P[se] - P[ss]
    return Network(
        node_x=P[:, 0], node_y=P[:, 1], node_z=P[:, 2], node_role=role,
        seg_start=ss, seg_end=se, seg_length=np.maximum(np.linalg.norm(d, axis=1), 1e-9),
        seg_radius=radius[se], seg_angle=np.arctan2(d[:, 1], d[:, 0]),
        root_node=int(base.root_node), inlet_nodes=base.inlet_nodes,
    )


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
    cand = _check_points(base, candidates, "candidates")
    dim = cand.shape[1]

    base_pos = _positions(base)
    base_par = _parent_nodes(base)
    n = len(base_pos)
    # Preallocated arrays (each added terminal adds at most two nodes), maintained
    # incrementally — no per-step list -> array rebuilds.
    cap = n + 2 * int(n_terminals)
    P = np.zeros((cap, dim))
    P[:n] = base_pos
    par = np.full(cap, -1, dtype=int)
    par[:n] = base_par
    term_idx = base.terminal_nodes
    terminals = np.zeros((max(len(term_idx), 1) + int(n_terminals), dim))
    n_term = len(term_idx) or 1
    terminals[:n_term] = base_pos[term_idx] if len(term_idx) else base_pos[base.root_node]

    if d_min is None:          # half the characteristic spacing (area or volume)^(1/dim)
        size = max(float(np.prod(np.ptp(cand, axis=0))), 1.0)
        d_min = 0.5 * (size / max(n_terminals, 1)) ** (1.0 / dim)

    rng = np.random.default_rng(seed)
    added = 0
    for idx in rng.permutation(len(cand)):
        if added >= n_terminals:
            break
        p = cand[idx]
        if np.min(np.linalg.norm(terminals[:n_term] - p, axis=1)) < d_min:
            continue                                   # too close to a terminal

        child = np.flatnonzero(par[:n] >= 0)
        if child.size == 0:                            # base is a lone node
            new_parent = base.root_node
        else:
            a, b = P[par[child]], P[child]
            ab = b - a
            denom = np.einsum("ij,ij->i", ab, ab)
            s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1), 0, 1)
            m = a + s[:, None] * ab
            j = int(np.argmin(np.linalg.norm(m - p, axis=1)))
            v, pu, sj = int(child[j]), int(par[child[j]]), float(s[j])
            if sj <= 1e-6:
                new_parent = pu
            elif sj >= 1 - 1e-6:
                new_parent = v
            else:                                       # split pu->v at the foot m_j
                new_parent = n
                P[n], par[n] = m[j], pu
                par[v] = new_parent
                n += 1
        P[n], par[n] = p, new_parent
        n += 1
        terminals[n_term] = p
        n_term += 1
        added += 1

    return _assemble(base, P[:n].tolist(), par[:n].tolist(), exponent=exponent,
                     leaf_radius=leaf_radius)


def cco_grow_in_mask(base: Network, mask, domain_limit, *, n_terminals=200,
                     n_pool=None, seed=0, density=None, obstacles=None, **kwargs) -> Network:
    """Convenience: sample a candidate pool in ``mask`` then :func:`cco_grow`.

    ``density`` (an ``(nx, ny)`` target map) biases the candidate pool toward
    high-demand tissue; ``obstacles`` (no-grow polygons) are excluded from it."""
    mask = mask_without_obstacles(mask, domain_limit, obstacles)
    pool = sample_points_in_mask(mask, domain_limit, n_pool or 6 * n_terminals,
                                 seed=seed, density=density)
    return cco_grow(base, pool, n_terminals=n_terminals, seed=seed, **kwargs)


def mask_without_obstacles(mask, domain_limit, obstacles) -> np.ndarray:
    """Carve ``obstacles`` (a list of no-grow polygons) out of ``mask``.

    Cells whose centre falls inside any obstacle polygon are deactivated, so growth
    (attraction points and terminals) avoids those regions — domain-aware growth
    around organ sub-structures / obstacles without editing the organ outline.
    """
    mask = np.asarray(mask, dtype=bool)
    if not obstacles:
        return mask
    from .core.geometry import rasterize_mask
    blocked = rasterize_mask(mask.shape[0], mask.shape[1], domain_limit, obstacles)
    return mask & ~blocked


def grow_in_mask(base: Network, mask, domain_limit, *, n_points=400, seed=0,
                 density=None, obstacles=None, **kwargs) -> Network:
    """Convenience: sample attraction points in ``mask`` then :func:`grow_network`.

    Pass ``density`` (an ``(nx, ny)`` target map) to bias the attraction points toward
    high-demand tissue (data-constrained growth), or ``obstacles`` (a list of no-grow
    polygons) to keep growth out of those regions. Any space-colonisation keyword
    (``step``, ``influence_radius``, …) is forwarded to :func:`grow_network`.
    """
    mask = mask_without_obstacles(mask, domain_limit, obstacles)
    pts = sample_points_in_mask(mask, domain_limit, n_points, seed=seed, density=density)
    return grow_network(base, pts, **kwargs)
