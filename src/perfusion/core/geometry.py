"""Vascular network representation and fractal-tree generation.

A :class:`Network` stores nodes (coordinates + role) and segments (parent/child
node, length, radius, angle) as parallel NumPy arrays, mirroring the MATLAB
``node``/``segment`` structs but with 0-based indexing.

:func:`grow_tree` reproduces ``Tree_artery.m`` / ``Tree_vein.m``: a self-similar
binary tree in which each generation's child radii and lengths scale by the
Murray-law ratios and branch off at the optimal bifurcation angles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import scipy.sparse as sp


class NodeRole(IntEnum):
    """Role of a node, matching the MATLAB ``junction`` codes."""

    TERMINAL = 0      # leaf (connects to the microcirculation)
    BIFURCATION = 1   # interior branch point
    ROOT = 9          # inlet/outlet of the tree


@dataclass
class Network:
    """A vascular tree (arterial or venous) as parallel node/segment arrays."""

    node_x: np.ndarray            # node x-coordinates
    node_y: np.ndarray            # node y-coordinates
    node_role: np.ndarray         # NodeRole per node
    seg_start: np.ndarray         # start (parent-side) node index of each segment
    seg_end: np.ndarray           # end (child-side) node index of each segment
    seg_length: np.ndarray        # segment length
    seg_radius: np.ndarray        # segment radius
    seg_angle: np.ndarray         # segment angle w.r.t. the x-axis (rad)
    root_node: int = 0
    # Prescribed-pressure boundary nodes (arterial inlets / venous outlets). Blood
    # enters/leaves here rather than coupling to the Darcy bed. Defaults to the
    # single root, reproducing the original one-inlet/one-outlet model.
    inlet_nodes: np.ndarray | None = None
    # Optional z-coordinate: when present the network is 3-D. The solver is
    # coordinate-free (topology + seg_length/seg_radius only), so it is unchanged;
    # only terminal->grid mapping and viz use coordinates. None => 2-D.
    node_z: np.ndarray | None = None

    # derived, filled in __post_init__
    incoming_segment: np.ndarray = field(init=False)  # segment feeding each node (-1 at root)

    def __post_init__(self) -> None:
        self.incoming_segment = np.full(self.n_nodes, -1, dtype=int)
        self.incoming_segment[self.seg_end] = np.arange(self.n_segments)
        if self.inlet_nodes is None:
            self.inlet_nodes = np.array([self.root_node], dtype=int)
        else:
            self.inlet_nodes = np.unique(np.asarray(self.inlet_nodes, dtype=int))

    @property
    def is_3d(self) -> bool:
        return self.node_z is not None

    @property
    def n_nodes(self) -> int:
        return len(self.node_x)

    @property
    def n_segments(self) -> int:
        return len(self.seg_length)

    @property
    def node_degree(self) -> np.ndarray:
        """Number of segments incident to each node."""
        return np.bincount(np.concatenate([self.seg_start, self.seg_end]),
                           minlength=self.n_nodes)

    @property
    def terminal_nodes(self) -> np.ndarray:
        """Leaf nodes (degree 1) that couple to the Darcy bed, minus inlet/outlet nodes.

        Classified by degree, not role, so any leaf -- including the BFS root when
        it is *not* an inlet -- correctly becomes a Darcy terminal.
        """
        leaf = self.node_degree == 1
        leaf[self.inlet_nodes] = False
        return np.flatnonzero(leaf)

    @property
    def bifurcation_nodes(self) -> np.ndarray:
        """Interior nodes (degree >= 2) that need mass conservation, minus inlets."""
        interior = self.node_degree >= 2
        interior[self.inlet_nodes] = False
        return np.flatnonzero(interior)

    def node_components(self) -> np.ndarray:
        """Connected-component label per node (over the segment graph).

        Lets callers verify each separate feeding tree has its own inlet; a
        component with no prescribed-pressure node gives a singular block.
        """
        adjacency: dict[int, list[int]] = {}
        for a, b in zip(self.seg_start, self.seg_end, strict=True):
            adjacency.setdefault(int(a), []).append(int(b))
            adjacency.setdefault(int(b), []).append(int(a))
        label = np.full(self.n_nodes, -1, dtype=int)
        current = 0
        for start in range(self.n_nodes):
            if label[start] >= 0:
                continue
            label[start] = current
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adjacency.get(u, ()):
                    if label[v] < 0:
                        label[v] = current
                        stack.append(v)
            current += 1
        return label

    def resistance(self, viscosity: float | np.ndarray) -> np.ndarray:
        """Poiseuille hydraulic resistance per segment, R = 8 mu L / (pi r^4)."""
        return 8.0 * viscosity * self.seg_length / (np.pi * self.seg_radius**4)

    def incidence(self) -> sp.csr_matrix:
        """Signed segment-node incidence matrix ``S`` (n_segments x n_nodes).

        ``S[s, start] = +1`` and ``S[s, end] = -1``, so ``(S @ p)[s]`` is the
        pressure drop ``p_start - p_end`` along segment ``s`` and ``(S.T @ q)``
        is the net outflow at each node. This turns Poiseuille and mass
        conservation into single matrix products.
        """
        seg = np.arange(self.n_segments)
        data = np.concatenate([np.ones(self.n_segments), -np.ones(self.n_segments)])
        rows = np.concatenate([seg, seg])
        cols = np.concatenate([self.seg_start, self.seg_end])
        return sp.csr_matrix((data, (rows, cols)),
                             shape=(self.n_segments, self.n_nodes))


def grow_tree(root_xy: tuple[float, float], root_angle: float, root_radius: float,
              min_radius: float, length_to_radius: float,
              angles: np.ndarray, radius_ratios: np.ndarray,
              mirror_children: bool = False) -> Network:
    """Grow a self-similar binary fractal tree.

    Starting from a trunk, every segment whose radius exceeds ``min_radius``
    splits into two children whose radii/lengths scale by ``radius_ratios`` and
    which deviate from the parent axis by ``angles`` (generation-indexed tables).

    Parameters
    ----------
    root_xy : root node coordinates.
    root_angle : trunk angle (rad).
    root_radius, min_radius : trunk radius; branching stops below ``min_radius``.
    length_to_radius : trunk length = ``length_to_radius * root_radius``.
    angles, radius_ratios : ``(n_generations, 2)`` tables of child angles and
        child/parent radius ratios (from Murray's law).
    mirror_children : swap the child ordering (arterial vs. venous convention in
        ``Tree_artery.m`` / ``Tree_vein.m``).

    Returns
    -------
    Network
    """
    n_generations = len(radius_ratios)

    # Growing lists; converted to arrays at the end.
    xs = [float(root_xy[0])]
    ys = [float(root_xy[1])]
    roles = [NodeRole.ROOT]
    node_incoming: list[int] = [-1]        # segment feeding each node (avoids O(n^2) search)

    seg_start = [0]
    seg_end = [1]
    seg_length = [length_to_radius * root_radius]
    seg_radius = [root_radius]
    seg_angle = [root_angle]

    xs.append(xs[0] + seg_length[0] * np.cos(root_angle))
    ys.append(ys[0] + seg_length[0] * np.sin(root_angle))
    roles.append(NodeRole.TERMINAL)        # provisional; set when the node is visited
    node_incoming.append(0)

    n_bifurcations = 0
    node = 1
    while node < len(xs):
        parent_seg = node_incoming[node]
        if seg_radius[parent_seg] > min_radius:
            roles[node] = NodeRole.BIFURCATION
            n_bifurcations += 1
            generation = min(n_bifurcations, n_generations) - 1
            # Two children; ``child`` in {0, 1} selects the ratio/angle column and
            # the sign of the angular deviation. ``mirror_children`` swaps them.
            for child in range(2):
                col = (1 - child) if mirror_children else child
                sign = -1.0 if child == 0 else +1.0
                ratio = radius_ratios[generation, col]
                angle = seg_angle[parent_seg] + sign * angles[generation, col]
                length = seg_length[parent_seg] * ratio
                radius = seg_radius[parent_seg] * ratio
                new_node = len(xs)
                seg_start.append(node)
                seg_end.append(new_node)
                seg_length.append(length)
                seg_radius.append(radius)
                seg_angle.append(angle)
                xs.append(xs[node] + length * np.cos(angle))
                ys.append(ys[node] + length * np.sin(angle))
                roles.append(NodeRole.TERMINAL)
                node_incoming.append(len(seg_start) - 1)
        else:
            roles[node] = NodeRole.TERMINAL
        node += 1

    return Network(
        node_x=np.asarray(xs), node_y=np.asarray(ys),
        node_role=np.asarray(roles, dtype=int),
        seg_start=np.asarray(seg_start, dtype=int),
        seg_end=np.asarray(seg_end, dtype=int),
        seg_length=np.asarray(seg_length), seg_radius=np.asarray(seg_radius),
        seg_angle=np.asarray(seg_angle), root_node=0,
    )


def bounding_box(*networks: Network, margin: float = 0.05
                 ) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box ``(xmin, xmax, ymin, ymax)`` of one or more trees.

    ``margin`` pads each side by that fraction of the larger extent, so a Darcy
    grid built from the box comfortably contains every terminal.
    """
    xs = np.concatenate([n.node_x for n in networks])
    ys = np.concatenate([n.node_y for n in networks])
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    pad = margin * max(xmax - xmin, ymax - ymin)
    return float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad)


def network_from_edges(node_x, node_y, edges, *, root: int | None = None,
                       root_xy: tuple[float, float] | None = None,
                       node_radius=None) -> Network:
    """Assemble a rooted :class:`Network` from a node list and edge list.

    ``edges`` is a list of ``(node_a, node_b, length, radius)``. Edges are
    oriented away from a root by a breadth-first search and node roles are set by
    degree (1 -> TERMINAL, >=2 -> BIFURCATION, root -> ROOT). The primary root is
    chosen as ``root`` if given, else the node nearest ``root_xy``, else the
    thickest leaf (by ``node_radius``).

    **Multiple connected components are supported**: each component is oriented
    from its own root (thickest leaf, or the primary root for the component that
    contains it), and every component root is added to ``inlet_nodes`` so the
    result is solvable (each separate feeding tree gets its own inlet).
    """
    node_x = np.asarray(node_x, dtype=float)
    node_y = np.asarray(node_y, dtype=float)
    n = len(node_x)

    degree = np.zeros(n, dtype=int)
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for e, (a, b, _, _) in enumerate(edges):
        degree[a] += 1
        degree[b] += 1
        adjacency.setdefault(a, []).append((b, e))
        adjacency.setdefault(b, []).append((a, e))

    if root is None:
        if root_xy is not None:
            root = int(np.argmin((node_x - root_xy[0]) ** 2 + (node_y - root_xy[1]) ** 2))
        else:
            leaves = np.flatnonzero(degree == 1)
            if leaves.size and node_radius is not None:
                root = int(leaves[np.argmax(np.asarray(node_radius)[leaves])])
            elif leaves.size:
                root = int(leaves[0])
            else:
                root = 0

    def _component_root(comp: list[int]) -> int:
        """Pick the root for one component: the primary root if it is in here,
        else the thickest leaf, else the first node."""
        if root in comp:
            return root
        leaves = [u for u in comp if degree[u] == 1]
        if leaves and node_radius is not None:
            rad = np.asarray(node_radius)
            return int(max(leaves, key=lambda u: rad[u]))
        return int(leaves[0]) if leaves else int(min(comp))

    # BFS-orient each connected component from its own root; collect the roots.
    order: dict[int, int] = {}
    roots: list[int] = []
    for seed in [root, *range(n)]:                     # primary component first
        if seed in order or seed >= n:
            continue
        comp, stack = [], [seed]                        # gather this component
        seen = {seed}
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adjacency.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        r = _component_root(comp)
        roots.append(r)
        order[r] = len(order)                           # BFS order from the root
        queue = deque([r])
        while queue:
            u = queue.popleft()
            for v, _ in adjacency.get(u, ()):
                if v not in order:
                    order[v] = len(order)
                    queue.append(v)

    seg_start, seg_end, seg_len, seg_rad, seg_theta = [], [], [], [], []
    for a, b, length, radius in edges:
        u, v = (a, b) if order.get(a, 1 << 30) <= order.get(b, 1 << 30) else (b, a)
        seg_start.append(u)
        seg_end.append(v)
        seg_len.append(length)
        seg_rad.append(radius)
        seg_theta.append(np.arctan2(node_y[v] - node_y[u], node_x[v] - node_x[u]))

    role = np.where(degree == 1, NodeRole.TERMINAL, NodeRole.BIFURCATION)
    role[roots] = NodeRole.ROOT
    return Network(
        node_x=node_x, node_y=node_y, node_role=role.astype(int),
        seg_start=np.asarray(seg_start, dtype=int),
        seg_end=np.asarray(seg_end, dtype=int),
        seg_length=np.asarray(seg_len, dtype=float),
        seg_radius=np.asarray(seg_rad, dtype=float),
        seg_angle=np.asarray(seg_theta, dtype=float), root_node=int(root),
        inlet_nodes=np.asarray(roots, dtype=int),
    )


def network_from_polylines(polylines, radii=1.0, root_xy: tuple[float, float] | None = None,
                           snap_tol: float = 1.0) -> Network:
    """Build a :class:`Network` from hand-drawn / edited vessel polylines.

    Each polyline is an ``(N, 2)`` sequence of points; its two ends become nodes
    (endpoints snapped together within ``snap_tol`` form shared junctions) and it
    becomes one segment whose length is the polyline arc length. ``radii`` is a
    scalar or one value per polyline. Powers the "draw vessels by hand" workflow.
    """
    coords: list[tuple[float, float]] = []
    index: dict[tuple[int, int], int] = {}

    def node_id(point) -> int:
        key = (round(point[0] / snap_tol), round(point[1] / snap_tol))
        if key not in index:
            index[key] = len(coords)
            coords.append((float(point[0]), float(point[1])))
        return index[key]

    if np.isscalar(radii):
        radii = [radii] * len(polylines)
    edges = []
    for polyline, radius in zip(polylines, radii, strict=False):
        pl = np.asarray(polyline, dtype=float)
        if len(pl) < 2:
            continue
        a, b = node_id(pl[0]), node_id(pl[-1])
        length = float(np.sum(np.hypot(np.diff(pl[:, 0]), np.diff(pl[:, 1]))))
        if a != b and length > 0:
            edges.append((a, b, length, float(radius)))
    if not edges:
        raise ValueError("need at least one polyline with two distinct endpoints")

    node_x = np.array([c[0] for c in coords])
    node_y = np.array([c[1] for c in coords])
    node_radius = np.zeros(len(coords))
    for a, b, _, r in edges:
        node_radius[a] = max(node_radius[a], r)
        node_radius[b] = max(node_radius[b], r)
    return network_from_edges(node_x, node_y, edges, root_xy=root_xy,
                              node_radius=node_radius)


def arterial_tree(prm) -> Network:
    """Build the arterial tree from :class:`~perfusion.parameters.Parameters`."""
    return grow_tree(prm.a_init, prm.a_theta, prm.r0, prm.r_min,
                     prm.length_to_radius, prm.phi, prm.ratio,
                     mirror_children=True)


def venous_tree(prm) -> Network:
    """Build the venous tree from :class:`~perfusion.parameters.Parameters`."""
    return grow_tree(prm.v_init, prm.v_theta, prm.rv0, prm.rv_min,
                     prm.length_to_radius, prm.vphi, prm.vratio,
                     mirror_children=False)


def _points_in_polygon(px: np.ndarray, py: np.ndarray, polygon) -> np.ndarray:
    """Even-odd ray-casting test, vectorised over the query points ``(px, py)``."""
    poly = np.asarray(polygon, dtype=float)
    if poly.shape[0] < 3:
        return np.zeros(px.shape, dtype=bool)
    x, y = poly[:, 0], poly[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        crosses = (y[i] > py) != (y[j] > py)
        # x-coordinate of the edge at height py (guard against a horizontal edge)
        x_at = (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-30) + x[i]
        inside ^= crosses & (px < x_at)
        j = i
    return inside


def rasterize_mask(nx: int, ny: int,
                   domain_limit: tuple[float, float, float, float],
                   polygons) -> np.ndarray:
    """Rasterise tissue outline ``polygons`` onto a ``(nx, ny)`` grid mask.

    Each polygon is a list of ``(x, y)`` vertices in the ``domain_limit``
    (``xmin, xmax, ymin, ymax``) frame; a cell is active if its centre lies
    inside the union of the polygons. With no polygons the whole grid is active
    (the bounding-box default). Cell centres match the grid convention used by
    ``run_hybrid`` (``lx = xmax - xmin``), so the mask aligns with the network
    coordinates.
    """
    if not polygons:
        return np.ones((nx, ny), dtype=bool)
    xmin, xmax, ymin, ymax = domain_limit
    xc = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    yc = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    gx, gy = np.meshgrid(xc, yc, indexing="ij")   # (nx, ny), column-major friendly
    px, py = gx.ravel(), gy.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    for poly in polygons:
        inside |= _points_in_polygon(px, py, poly)
    return inside.reshape(nx, ny)


def rasterize_permeability(nx: int, ny: int,
                           domain_limit: tuple[float, float, float, float],
                           lesions, base: float = 1.0) -> np.ndarray:
    """Rasterise a heterogeneous permeability field from ``lesions`` onto ``(nx, ny)``.

    Starts from a uniform ``base`` permeability; each lesion is ``(polygon, k)`` (a
    list of ``(x, y)`` vertices in the ``domain_limit`` frame and its permeability),
    and every cell whose centre falls inside overwrites to that ``k`` (later lesions
    win on overlap). With no lesions the field is uniform ``base``. Cell centres match
    :func:`rasterize_mask` so the field aligns with the mask and network coordinates.
    """
    field = np.full((nx, ny), float(base))
    if not lesions:
        return field
    xmin, xmax, ymin, ymax = domain_limit
    xc = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    yc = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    flat = field.ravel()
    for poly, k in lesions:
        inside = _points_in_polygon(px, py, poly)
        flat[inside] = float(k)
    return flat.reshape(nx, ny)


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``vec`` about unit ``axis`` by ``angle``."""
    c, s = np.cos(angle), np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1 - c)


def grow_tree_3d(root_xyz, direction=(0.0, 0.0, 1.0), *, root_radius=2.0,
                 min_radius=0.6, length_to_radius=8.0, theta=0.5, ratio=0.78,
                 max_generations=8, mirror=False) -> Network:
    """Grow a 3-D self-similar binary fractal tree.

    Every segment above ``min_radius`` splits into two children rotated by
    ``±theta`` from the parent direction; the branching plane alternates each
    generation (about z then y) so the tree fills 3-D. Radii/lengths scale by
    ``ratio``. Returns a :class:`Network` carrying ``node_z`` (a 3-D network).
    The solver is coordinate-free, so this plugs into the same hybrid solve.
    """
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    root = np.asarray(root_xyz, dtype=float)
    d0 = unit(np.asarray(direction, dtype=float))
    xs, ys, zs, roles = [root[0]], [root[1]], [root[2]], [NodeRole.ROOT]
    seg_start, seg_end, seg_len, seg_rad, seg_ang = [], [], [], [], []

    def add_node(p):
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        roles.append(NodeRole.TERMINAL)
        return len(xs) - 1

    def add_seg(parent, child, length, radius):
        seg_start.append(parent)
        seg_end.append(child)
        seg_len.append(length)
        seg_rad.append(radius)
        seg_ang.append(0.0)

    L0 = length_to_radius * root_radius
    add_seg(0, add_node(root + d0 * L0), L0, root_radius)

    axes = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    queue = deque([(1, d0, root_radius, L0, 0)])
    while queue:
        node, d, r, length, gen = queue.popleft()
        if r <= min_radius or gen >= max_generations:
            continue
        roles[node] = NodeRole.BIFURCATION
        axis = np.cross(axes[gen % 2], d)
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(axes[(gen + 1) % 2], d)
        axis = unit(axis)
        cr, cl = r * ratio, length * ratio
        parent_pos = np.array([xs[node], ys[node], zs[node]])
        for sign in ((-1.0, 1.0) if not mirror else (1.0, -1.0)):
            child_dir = unit(_rotate(d, axis, sign * theta))
            child = add_node(parent_pos + child_dir * cl)
            add_seg(node, child, cl, cr)
            queue.append((child, child_dir, cr, cl, gen + 1))

    return Network(
        node_x=np.asarray(xs), node_y=np.asarray(ys),
        node_role=np.asarray(roles, dtype=int),
        seg_start=np.asarray(seg_start, dtype=int), seg_end=np.asarray(seg_end, dtype=int),
        seg_length=np.asarray(seg_len), seg_radius=np.asarray(seg_rad),
        seg_angle=np.asarray(seg_ang), root_node=0, node_z=np.asarray(zs),
    )


def _subnetwork(net: Network, keep_seg: np.ndarray) -> Network:
    """The network restricted to ``keep_seg`` segments; nodes left with no segment
    are dropped unless they are inlets. Reindexes nodes, re-derives roles by degree,
    and carries ``inlet_nodes``/``root_node``/``node_z`` across."""
    ss, se = net.seg_start[keep_seg], net.seg_end[keep_seg]
    keep_node = np.zeros(net.n_nodes, bool)
    keep_node[ss] = True
    keep_node[se] = True
    keep_node[net.inlet_nodes] = True
    old2new = np.full(net.n_nodes, -1, dtype=int)
    old2new[keep_node] = np.arange(int(keep_node.sum()))
    ss2, se2 = old2new[ss], old2new[se]
    nx, ny = net.node_x[keep_node], net.node_y[keep_node]
    inlets = old2new[net.inlet_nodes]
    deg = np.bincount(np.concatenate([ss2, se2]), minlength=len(nx))
    role = np.where(deg >= 2, int(NodeRole.BIFURCATION), int(NodeRole.TERMINAL))
    role[inlets] = int(NodeRole.ROOT)
    root = int(old2new[net.root_node]) if keep_node[net.root_node] else int(inlets[0])
    return Network(
        node_x=nx, node_y=ny, node_role=role, seg_start=ss2, seg_end=se2,
        seg_length=net.seg_length[keep_seg], seg_radius=net.seg_radius[keep_seg],
        seg_angle=net.seg_angle[keep_seg], root_node=root, inlet_nodes=inlets,
        node_z=net.node_z[keep_node] if net.node_z is not None else None,
    )


def prune_terminal_branches(net: Network, steps: int = 1) -> Network:
    """Prune the outermost ``steps`` generations of a tree — the inverse of growth.

    One step removes every *terminal branch*: the chain of segments from a free end
    (a degree-1 node that is not an inlet/outlet) back through pass-through
    (degree-2) nodes to the nearest junction. All terminal branches of a generation
    go at once, so a junction whose children were all pruned becomes a free end
    that the next step removes. A chain that reaches an inlet/outlet (the trunk) is
    never removed, so the fed tree survives; loops are untouched (no free end).
    Segment geometry and radii of what remains are unchanged.
    """
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be >= 0")
    for _ in range(steps):
        if net.n_segments == 0:
            break
        deg = net.node_degree
        inlet = np.zeros(net.n_nodes, bool)
        inlet[net.inlet_nodes] = True
        adj: list[list[tuple[int, int]]] = [[] for _ in range(net.n_nodes)]
        for s, (a, b) in enumerate(zip(net.seg_start, net.seg_end, strict=True)):
            adj[a].append((int(b), s))
            adj[b].append((int(a), s))
        drop = np.zeros(net.n_segments, bool)
        for leaf in np.flatnonzero((deg == 1) & ~inlet):
            chain, prev, cur = [], -1, int(leaf)
            while True:
                nxt = [(v, s) for v, s in adj[cur] if s not in chain and v != prev]
                if not nxt:                           # an isolated chain end: whole fragment
                    break
                v, s = nxt[0]
                chain.append(s)
                prev, cur = cur, v
                if inlet[cur] or deg[cur] != 2:       # junction, trunk or the far end
                    break
            if not inlet[cur]:                        # never cut the fed trunk
                drop[chain] = True
        if not drop.any():
            break
        net = _subnetwork(net, ~drop)
    return net
