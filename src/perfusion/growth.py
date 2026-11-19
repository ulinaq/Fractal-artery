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
                 leaf_radius=None, close_loops: bool = False, fuse_radius=None,
                 min_detour: float = 3.0, min_hops: int = 4,
                 loop_radius_frac: float = 0.7, max_loops: int | None = None) -> Network:
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
    close_loops, fuse_radius, min_detour, min_hops, loop_radius_frac, max_loops :
        **Loop closure during growth** (inosculation): a freshly grown tip that comes
        within ``fuse_radius`` (default ``2 × step``) of a node on a *different*
        branch fuses with it — a new segment joins them and the tip stops growing.
        "Different branch" means the tree path between them is at least
        ``min_detour`` × their distance and ``min_hops`` segments long (so a tip never
        fuses back onto its own stem). The fusing segment's radius is
        ``loop_radius_frac`` × the thinner of the two vessels (after Murray tapering);
        at most ``max_loops`` fusions. Inlets/roots are never fused onto. Loops
        already present in ``base`` are kept either way.

    Returns
    -------
    Network
        A new network (base topology + grown branches), rooted at ``base``'s
        root, with Murray-tapered radii (plus any loop-closing segments).
    """
    pts = _check_points(base, attraction_points, "attraction_points")

    # Node arrays, seeded from the base network (2-D or 3-D).
    pos = _positions(base).tolist()
    parent_arr, extra = _spanning_parents(base)
    parent = parent_arr.tolist()
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

    if fuse_radius is None:
        fuse_radius = 2.0 * step
    fuse = _Fuser(parent, pos, base, float(fuse_radius), float(min_detour), int(min_hops),
                  max_loops) if close_loops else None

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
        new_nodes = []
        occupied = cKDTree(node_arr)
        for node, direction in pull.items():
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            new_xy = node_arr[node] + step * direction / norm
            # a node pulled from opposite sides would re-sprout onto the same spot every
            # iteration (its child is never nearer those points): skip occupied spots
            if occupied.query(new_xy)[0] < 0.5 * step or any(
                    np.linalg.norm(np.subtract(pos[m], new_xy)) < 0.5 * step
                    for m in new_nodes):
                continue
            pos.append([float(c) for c in new_xy])
            parent.append(node)
            eligible.append(True)
            new_nodes.append(len(pos) - 1)
            grew = True
        if not grew:
            break
        if fuse is not None:                 # tips meeting another branch fuse (loops)
            for v in fuse.check(new_nodes, node_arr):
                eligible[v] = False

    loops = [(a, b, None) for a, b in (fuse.links if fuse else [])]
    return _assemble(base, pos, parent, exponent=exponent, leaf_radius=leaf_radius,
                     extra=extra + loops, loop_radius_frac=loop_radius_frac)


def _spanning_parents(base: Network):
    """Parent pointers of a spanning tree of ``base`` from its root, plus the edges
    left over (loops already in ``base``) as ``[(a, b, radius)]`` so they survive.

    The tree is a *maximum* spanning tree by radius (widest vessels first), so in a
    looped network the thin collaterals are the edges set aside — the trunk keeps its
    Murray taper and each collateral its own radius."""
    from collections import deque

    n = base.n_nodes
    root = list(range(n))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    used = np.zeros(base.n_segments, dtype=bool)
    for s in np.argsort(-np.asarray(base.seg_radius, float), kind="stable"):
        a, b = find(int(base.seg_start[s])), find(int(base.seg_end[s]))
        if a != b:
            root[a] = b
            used[s] = True
    adj: dict[int, list[int]] = {}
    for s in np.flatnonzero(used):
        a, b = int(base.seg_start[s]), int(base.seg_end[s])
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    parent = np.full(n, -1, dtype=int)
    seen = np.zeros(n, dtype=bool)
    starts = [int(base.root_node), *map(int, np.asarray(base.inlet_nodes)), *range(n)]
    for r in starts:                         # orient each component from its root
        if seen[r]:
            continue
        seen[r] = True
        queue = deque([r])
        while queue:
            u = queue.popleft()
            for v in adj.get(u, ()):
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    queue.append(v)
    extra = [(int(base.seg_start[s]), int(base.seg_end[s]), float(base.seg_radius[s]))
             for s in np.flatnonzero(~used)]
    return parent, extra


class _Fuser:
    """Loop closure during colonisation: decide which new tips fuse with a nearby
    node on another branch (tree-path detour + hop criteria) and record the links."""

    def __init__(self, parent, pos, base, radius, min_detour, min_hops, max_loops):
        self.parent, self.pos = parent, pos
        self.radius, self.min_detour, self.min_hops = radius, min_detour, min_hops
        self.max_loops = max_loops
        self.forbidden = {int(base.root_node), *map(int, np.asarray(base.inlet_nodes))}
        self.links: list[tuple[int, int]] = []
        self.fused: set[int] = set()
        self.dlen: list[float] = []          # path length from the root (spanning tree)
        self.hops: list[int] = []            # segments from the root

    def _depth(self, v):
        """Fill the cached root distances up to node ``v`` (parents precede children
        for grown nodes; base nodes are resolved by walking up)."""
        while len(self.dlen) <= v:
            self.dlen.append(np.nan)
            self.hops.append(-1)
        if self.hops[v] >= 0:
            return
        chain = []
        while v >= 0 and self.hops[v] < 0:
            chain.append(v)
            v = self.parent[v]
        for w in reversed(chain):
            p = self.parent[w]
            if p < 0:
                self.dlen[w], self.hops[w] = 0.0, 0
            else:
                self.dlen[w] = self.dlen[p] + float(
                    np.linalg.norm(np.subtract(self.pos[w], self.pos[p])))
                self.hops[w] = self.hops[p] + 1

    def _tree_path(self, v, u):
        """(length, hops) of the spanning-tree path between nodes ``v`` and ``u``."""
        self._depth(v)
        self._depth(u)
        a, b = v, u
        while self.hops[a] > self.hops[b]:
            a = self.parent[a]
        while self.hops[b] > self.hops[a]:
            b = self.parent[b]
        while a != b:
            a, b = self.parent[a], self.parent[b]
            if a < 0 or b < 0:                   # different components
                return np.inf, 1 << 30
        return (self.dlen[v] + self.dlen[u] - 2 * self.dlen[a],
                self.hops[v] + self.hops[u] - 2 * self.hops[a])

    def check(self, new_nodes, old_positions) -> list[int]:
        """Fuse each new tip with its nearest qualifying older node (not one grown in
        the same step). Returns the tips that fused (they stop growing)."""
        if self.max_loops is not None and len(self.links) >= self.max_loops:
            return []
        tree = cKDTree(old_positions)
        fused = []
        for v in new_nodes:
            hits = tree.query_ball_point(self.pos[v], self.radius)
            best = None
            for u in sorted(hits, key=lambda u: np.linalg.norm(
                    np.subtract(self.pos[v], old_positions[u]))):
                if u in self.forbidden or u == self.parent[v] or u in self.fused:
                    continue
                gap = float(np.linalg.norm(np.subtract(self.pos[v], old_positions[u])))
                length, hops = self._tree_path(v, u)
                if hops >= self.min_hops and length >= self.min_detour * max(gap, 1e-12):
                    best = u
                    break
            if best is None:
                continue
            self.links.append((v, int(best)))
            self.fused.update((v, int(best)))
            fused.append(v)
            if self.max_loops is not None and len(self.links) >= self.max_loops:
                break
        return fused


def _assemble(base, pos, parent, *, exponent, leaf_radius, extra=(), loop_radius_frac=0.7):
    """Turn the grown (positions, parent) tree into a rooted :class:`Network`;
    ``extra`` are non-tree edges ``(a, b, radius)`` — loops kept from the base, or
    fused during growth (radius ``None`` ⇒ ``loop_radius_frac`` × the thinner side)."""
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
    loops = [(int(a), int(b), float(r) if r is not None
              else loop_radius_frac * float(min(radius[a], radius[b])))
             for a, b, r in extra]

    if P.shape[1] == 3:
        return _assemble_3d(base, P, parent, radius, loops)
    edges = []
    for v in range(n_nodes):
        p = parent[v]
        if p >= 0:
            length = float(np.hypot(node_x[v] - node_x[p], node_y[v] - node_y[p]))
            edges.append((int(p), int(v), max(length, 1e-9), float(radius[v])))
    for a, b, r in loops:
        edges.append((a, b, max(float(np.hypot(*(P[b] - P[a]))), 1e-9), r))
    net = network_from_edges(node_x, node_y, edges, root=base.root_node)
    return _keep_inlets(net, base)


def _keep_inlets(net: Network, base: Network) -> Network:
    """Growth only appends nodes, so ``base``'s inlet nodes keep their indices."""
    from dataclasses import replace

    inlets = np.union1d(np.asarray(base.inlet_nodes, dtype=int),
                        np.asarray(net.inlet_nodes, dtype=int))
    if np.array_equal(np.sort(np.asarray(net.inlet_nodes)), inlets):
        return net
    return replace(net, inlet_nodes=inlets)


def _assemble_3d(base, P, parent, radius, loops=()):
    """3-D twin of the 2-D assembly: the grown tree is already parent→child
    oriented from ``base``'s root, so build the :class:`Network` directly (loop
    segments appended)."""
    from .core.geometry import NodeRole

    child = np.flatnonzero(parent >= 0)
    ss, se, rad = parent[child], child, radius[child]
    if loops:
        la = np.array([a for a, _, _ in loops], dtype=int)
        lb = np.array([b for _, b, _ in loops], dtype=int)
        ss, se = np.concatenate([ss, la]), np.concatenate([se, lb])
        rad = np.concatenate([rad, [r for _, _, r in loops]])
    deg = np.bincount(np.concatenate([ss, se]), minlength=len(P))
    role = np.where(deg >= 2, int(NodeRole.BIFURCATION), int(NodeRole.TERMINAL))
    role[base.root_node] = int(NodeRole.ROOT)
    d = P[se] - P[ss]
    return Network(
        node_x=P[:, 0], node_y=P[:, 1], node_z=P[:, 2], node_role=role,
        seg_start=ss, seg_end=se, seg_length=np.maximum(np.linalg.norm(d, axis=1), 1e-9),
        seg_radius=rad, seg_angle=np.arctan2(d[:, 1], d[:, 0]),
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
    # a spanning tree from the root (any segment orientation; loops in the base set
    # aside and re-added below) — not ``parent[seg_end] = seg_start``, which breaks
    # on a reversed hand-drawn segment and drops anastomoses
    base_par, extra = _spanning_parents(base)
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
                     leaf_radius=leaf_radius, extra=extra)


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
