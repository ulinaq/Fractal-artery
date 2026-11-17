"""Turn a segmented vessel image into a :class:`~perfusion.geometry.Network`.

Pipeline (mirrors the MATLAB ``Skeleton``/``seggg`` tooling):

1. **Threshold** the image to a binary vessel mask.
2. **Skeletonise** it to a 1-pixel-wide centreline.
3. **Trace** the skeleton into a graph: endpoints (degree 1) and junctions
   (degree >= 3) become nodes; the pixel paths between them become segments.
4. **Radius** per segment from the Euclidean distance transform of the mask
   (distance from centreline to the vessel wall).
5. **Root & orient** the graph by a breadth-first search so it plugs straight
   into the hybrid solver.

Requires the ``image`` extra (``uv sync --extra image``) for scikit-image.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

from .core.geometry import Network, network_from_edges

# 8-connectivity structuring element and neighbour offsets.
_STRUCT = np.ones((3, 3), dtype=int)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse an RGB(A) image to grayscale luminance; pass through if 2-D."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(float)
    rgb = image[..., :3].astype(float)
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def vessel_mask(image: np.ndarray, threshold: float | None = None,
                min_size: int = 0) -> np.ndarray:
    """Binary vessel mask from an image (vessels are the bright pixels).

    ``threshold`` defaults to Otsu's method. ``min_size`` removes connected
    components smaller than that many pixels (speckle).
    """
    gray = _to_gray(image)
    if threshold is None:
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(gray)
    mask = gray > threshold
    if min_size > 0:                          # drop connected speckle < min_size px
        labels, _ = ndi.label(mask, structure=_STRUCT)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = (sizes >= min_size)[labels]
    return mask


def _neighbours(r: int, c: int, shape: tuple[int, int]):
    """Yield in-bounds 8-neighbours of pixel (r, c)."""
    h, w = shape
    for dr, dc in _OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _trace_edges(skel, node_label, is_node, is_link, dist, pixel_size):
    """Walk each skeleton path once, returning edges between graph nodes.

    Each edge is ``(node_a, node_b, length, radius)`` with 0-based node ids.
    """
    visited = np.zeros_like(skel, dtype=bool)
    edges = []
    node_rc = np.argwhere(is_node)
    for r, c in node_rc:
        for nr, nc in _neighbours(r, c, skel.shape):
            if not is_link[nr, nc] or visited[nr, nc]:
                continue
            # walk the path of degree-2 link pixels until the next node
            path = [(r, c)]
            prev, cur = (r, c), (nr, nc)
            while is_link[cur]:
                visited[cur] = True
                path.append(cur)
                nxt = next(
                    (p for p in _neighbours(*cur, skel.shape)
                     if p != prev and skel[p] and (is_link[p] or is_node[p])),
                    None,
                )
                if nxt is None:
                    break
                prev, cur = cur, nxt
            path.append(cur)
            a, b = node_label[r, c] - 1, node_label[cur] - 1
            if a == b or a < 0 or b < 0:
                continue                      # self-loop or dangling
            rows, cols = zip(*path, strict=False)
            length = pixel_size * (len(path) - 1)
            radius = pixel_size * float(np.mean(dist[list(rows), list(cols)]))
            edges.append((a, b, length, max(radius, pixel_size)))
    return edges


def skeleton_to_network(mask: np.ndarray, root_xy: tuple[float, float] | None = None,
                        pixel_size: float = 1.0) -> Network:
    """Trace a binary vessel mask into a rooted :class:`Network`.

    Parameters
    ----------
    mask : boolean vessel mask.
    root_xy : coordinates of the inlet; the nearest node becomes the root. If
        ``None``, the thickest endpoint (largest radius) is used -- typically the
        feeding trunk.
    pixel_size : physical size of a pixel (sets length/radius units).
    """
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    skel = skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)

    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    degree = ndi.convolve(skel.astype(int), kernel, mode="constant") * skel
    is_node = skel & (degree != 2)            # endpoints + junctions
    is_link = skel & (degree == 2)            # interior path pixels

    node_label, n_nodes = ndi.label(is_node, structure=_STRUCT)
    if n_nodes == 0:
        raise ValueError("no vessel skeleton found; check the threshold")
    centroids = ndi.center_of_mass(np.ones_like(node_label), node_label,
                                   range(1, n_nodes + 1))
    node_x = pixel_size * np.array([r for r, _ in centroids])
    node_y = pixel_size * np.array([c for _, c in centroids])

    node_radius = pixel_size * np.array(
        [dist[round(r), round(c)] for r, c in centroids])

    edges = _trace_edges(skel, node_label, is_node, is_link, dist, pixel_size)
    if not edges:
        raise ValueError("skeleton produced no segments")

    # Keep the largest connected component, then renumber its nodes to 0..k-1
    # (isolated leftover nodes would otherwise make the solver singular).
    adjacency: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_nodes)}
    for e, (a, b, _, _) in enumerate(edges):
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    component = _largest_component(adjacency, n_nodes)
    keep = sorted(component)
    relabel = {old: new for new, old in enumerate(keep)}
    node_x, node_y, node_radius = node_x[keep], node_y[keep], node_radius[keep]
    edges = [(relabel[a], relabel[b], ln, rad) for a, b, ln, rad in edges
             if a in component and b in component]

    # Orient, root and assemble via the shared core builder.
    return network_from_edges(node_x, node_y, edges, root_xy=root_xy,
                              node_radius=node_radius)


def _largest_component(adjacency: dict[int, list], n_nodes: int) -> set[int]:
    """Return the node set of the largest connected component."""
    seen: set[int] = set()
    best: set[int] = set()
    for start in range(n_nodes):
        if start in seen or not adjacency[start]:
            continue
        comp, queue = set(), deque([start])
        while queue:
            u = queue.popleft()
            if u in comp:
                continue
            comp.add(u)
            queue.extend(v for v, _ in adjacency[u] if v not in comp)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return best


def network_in_domain(net: Network, image_shape: tuple[int, int],
                      domain_limit: tuple[float, float, float, float]) -> Network:
    """Place a pixel-space segmented ``net`` into a physical ``domain_limit`` box.

    :func:`skeleton_to_network` returns coordinates in image pixels
    (``node_x`` = row, ``node_y`` = column). This maps them so the image fills
    ``(xmin, xmax, ymin, ymax)`` the way it reads on screen: column -> x
    (left->right), row -> y (top->bottom, flipped so the image top is at
    ``ymax``). Segment lengths, angles and radii are rescaled to match.
    """
    h, w = image_shape[0], image_shape[1]
    xmin, xmax, ymin, ymax = domain_limit
    node_x = xmin + (net.node_y / max(w, 1)) * (xmax - xmin)
    node_y = ymin + (1.0 - net.node_x / max(h, 1)) * (ymax - ymin)
    ss, se = net.seg_start, net.seg_end
    length = np.hypot(node_x[se] - node_x[ss], node_y[se] - node_y[ss])
    angle = np.arctan2(node_y[se] - node_y[ss], node_x[se] - node_x[ss])
    scale = 0.5 * ((xmax - xmin) / max(w, 1) + (ymax - ymin) / max(h, 1))
    return replace(net, node_x=node_x, node_y=node_y, seg_length=length,
                   seg_radius=net.seg_radius * scale, seg_angle=angle)


def mask_in_domain(mask: np.ndarray, nx: int, ny: int, domain_limit) -> np.ndarray:
    """Resample a pixel mask onto an ``(nx, ny)`` grid over ``domain_limit``.

    Uses the same (column -> x, row -> y flipped) convention as
    :func:`network_in_domain`, so a segmented network and its organ mask stay
    aligned after both are placed in the domain. Nearest-neighbour sampling
    preserves connectivity (unlike a polygon-outline approximation).
    """
    mask = np.asarray(mask)
    h, w = mask.shape
    xmin, xmax, ymin, ymax = domain_limit
    xc = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    yc = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    col = np.clip(((gx - xmin) / (xmax - xmin) * w).astype(int), 0, w - 1)
    row = np.clip(((1 - (gy - ymin) / (ymax - ymin)) * h).astype(int), 0, h - 1)
    return mask[row, col]


def segment_network(image: np.ndarray, threshold: float | None = None,
                    root_xy: tuple[float, float] | None = None,
                    pixel_size: float = 1.0, min_size: int = 20) -> Network:
    """Segment a vessel image directly into a rooted :class:`Network`."""
    mask = vessel_mask(image, threshold=threshold, min_size=min_size)
    return skeleton_to_network(mask, root_xy=root_xy, pixel_size=pixel_size)


@dataclass
class FrogSystem:
    """The full segmented frog-tongue system: arteries, veins and organ domain."""

    artery: Network
    vein: Network
    mask: np.ndarray                 # (nx, ny) organ (tongue) mask
    grid: object                   # perfusion.darcy.Grid over the organ

    @property
    def domain_limit(self) -> tuple[float, float, float, float]:
        nx, ny = self.mask.shape
        return (0.0, float(nx), 0.0, float(ny))


def load_frog_system(min_size: int = 30) -> FrogSystem:
    """Segment the bundled frog images into a coupled arterial/venous system.

    Reads ``frog_art_low.png`` (arteries), ``frog_vein.png`` (veins) and
    ``frog_domain_low.png`` (the tongue outline) from ``data/``, segments the two
    vessel images into rooted networks, and builds a Darcy grid over the organ
    mask. The three share one pixel coordinate system, so the result plugs
    directly into :func:`perfusion.pipeline.run_hybrid`.

    Requires the ``image`` extra (``uv sync --extra image``).
    """
    import imageio.v3 as iio

    from .core.darcy import Grid
    from .io import DATA_DIR

    artery = segment_network(iio.imread(DATA_DIR / "frog_art_low.png"), min_size=min_size)
    vein = segment_network(iio.imread(DATA_DIR / "frog_vein.png"), min_size=min_size)

    dom = iio.imread(DATA_DIR / "frog_domain_low.png")
    mask = _to_gray(dom) > (_to_gray(dom).max() / 2)
    nx, ny = mask.shape
    grid = Grid(nx=nx, ny=ny, lx=nx, ly=ny, mask=mask)
    return FrogSystem(artery=artery, vein=vein, mask=mask, grid=grid)
