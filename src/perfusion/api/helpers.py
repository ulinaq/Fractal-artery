"""Compute helpers shared by the route modules (no routing, no state).

Kept out of the routers so the request handlers stay thin and these stay unit-
testable: the frog/demo 3-D scene builders, the edited-network rebuild, and the
tracer GIF / MP4 renderers.
"""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException

# The flow-video renderers live in the service layer (``perfusion.video``) so the MCP
# server can use them without the API's FastAPI dependency; re-exported here.
from ..video import project_3d as project_3d
from ..video import render_gif as render_gif
from ..video import render_mp4 as render_mp4
from ..video import segment_lines_3d as segment_lines_3d
from ..video import tracer_images_3d as tracer_images_3d

# result-field name -> stored-array key
FIELDS = {"pressure": "pressure", "flux": "flux", "tof": "tof"}

_DEMO3D_CACHE: dict = {}


def segment_lines(net):
    """Yield each segment as ``(x0, y0, x1, y1, radius)`` in domain coords (2-D).

    One place for the ``seg_start``/``seg_end`` → node-coordinate lookup that the
    /network, /view3d and frog-demo serialisers all need.
    """
    for s in range(net.n_segments):
        a, b = int(net.seg_start[s]), int(net.seg_end[s])
        yield (float(net.node_x[a]), float(net.node_y[a]),
               float(net.node_x[b]), float(net.node_y[b]), float(net.seg_radius[s]))


def frog_networks_and_mask(nx: int, ny: int, dom):
    """Segment the bundled frog system into (artery, vein, raster mask) placed in
    ``dom``. Raises ImportError without the ``image`` extra."""
    from .. import segmentation as seg

    system = seg.load_frog_system()
    return (seg.network_in_domain(system.artery, system.mask.shape, dom),
            seg.network_in_domain(system.vein, system.mask.shape, dom),
            seg.mask_in_domain(system.mask, nx, ny, dom))


def demo3d_frog() -> dict:
    """The 2-D frog result as a 3-D pressure-relief surface (z = pressure) with
    the vasculature at the base — for the 3-D viewer."""
    if "frog" in _DEMO3D_CACHE:
        return _DEMO3D_CACHE["frog"]
    from ..core.darcy import Grid
    from ..core.parameters import Parameters
    from ..core.pipeline import run_hybrid

    dom = (0.0, 100.0, 0.0, 120.0)
    nx, ny = 120, 140
    art, ven, mask = frog_networks_and_mask(nx, ny, dom)
    grid = Grid(nx=nx, ny=ny, lx=100.0, ly=120.0, mask=mask)
    res = run_hybrid(Parameters(p_in=40.0, p_out=0.0), grid, domain_limit=dom, art=art, ven=ven)
    p = res.pressure
    finite = p[np.isfinite(p) & mask]
    pmin, pmax = float(finite.min()), float(finite.max())
    span = (pmax - pmin) or 1.0
    relief = 35.0
    xc = (np.arange(nx) + 0.5) / nx * 100.0
    yc = (np.arange(ny) + 0.5) / ny * 120.0
    stride = max(1, round((nx * ny / 2500) ** 0.5))
    voxels = [
        [round(float(xc[ix]), 1), round(float(yc[iy]), 1),
         round((p[ix, iy] - pmin) / span * relief, 2), round(float(p[ix, iy]), 4)]
        for ix in range(0, nx, stride) for iy in range(0, ny, stride)
        if mask[ix, iy] and np.isfinite(p[ix, iy])
    ]

    def segs(net, kind):
        k = 0 if kind == "artery" else 1
        return [[round(x0, 2), round(y0, 2), 0.0, round(x1, 2), round(y1, 2), 0.0, round(r, 2), k]
                for x0, y0, x1, y1, r in segment_lines(net)]

    payload = {"domain": [0.0, 100.0, 0.0, 120.0, 0.0, relief], "shape": [nx, ny, 1],
               "pressure": {"min": pmin, "max": pmax, "voxels": voxels},
               "segments": segs(art, "artery") + segs(ven, "vein")}
    _DEMO3D_CACHE["frog"] = payload
    return payload


def demo3d_payload(gen: int = 6) -> dict:
    """Run a default 3-D hybrid and serialise trees + a subsampled voxel field.

    Cached per ``gen``. Feeds the browser 3-D viewer (``frontend/viewer3d.js``).
    """
    gen = int(np.clip(gen, 3, 7))
    if gen in _DEMO3D_CACHE:
        return _DEMO3D_CACHE[gen]

    from ..core.darcy3d import Grid3D
    from ..core.geometry import grow_tree_3d
    from ..core.parameters import Parameters
    from ..core.pipeline3d import run_hybrid_3d

    art = grow_tree_3d((25, 25, 3), (0, 0, 1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=gen)
    ven = grow_tree_3d((25, 25, 47), (0, 0, -1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=gen, mirror=True)
    xs = np.concatenate([art.node_x, ven.node_x])
    ys = np.concatenate([art.node_y, ven.node_y])
    zs = np.concatenate([art.node_z, ven.node_z])
    m = 4.0
    dom = (xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m, zs.min() - m, zs.max() + m)
    grid = Grid3D(nx=26, ny=26, nz=28,
                  lx=dom[1] - dom[0], ly=dom[3] - dom[2], lz=dom[5] - dom[4])
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)

    def segments(net, kind):
        k = 0 if kind == "artery" else 1
        return [[round(x0, 2), round(y0, 2), round(z0, 2), round(x1, 2), round(y1, 2),
                 round(z1, 2), round(r, 2), k]
                for x0, y0, z0, x1, y1, z1, r in segment_lines_3d(net)]

    p, mask = res.pressure, grid.mask
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    finite = p[np.isfinite(p) & mask]
    pmin, pmax = float(finite.min()), float(finite.max())
    stride = max(1, round((nx * ny * nz / 3000) ** (1 / 3)))
    xmin, _, ymin, _, zmin, _ = dom
    voxels = [
        [round(xmin + (ix + 0.5) * grid.hx, 1), round(ymin + (iy + 0.5) * grid.hy, 1),
         round(zmin + (iz + 0.5) * grid.hz, 1), round(float(p[ix, iy, iz]), 4)]
        for ix in range(0, nx, stride)
        for iy in range(0, ny, stride)
        for iz in range(0, nz, stride)
        if mask[ix, iy, iz] and np.isfinite(p[ix, iy, iz])
    ]
    payload = {
        "domain": [round(v, 1) for v in dom], "shape": [nx, ny, nz],
        "pressure": {"min": pmin, "max": pmax, "voxels": voxels},
        "segments": segments(art, "artery") + segments(ven, "vein"),
    }
    _DEMO3D_CACHE[gen] = payload
    return payload


def rebuild_network(template, node_x, node_y, seg_start, seg_end, seg_radius,
                    inlet_nodes, root_node, node_z=None):
    """Rebuild a :class:`Network` from edited nodes + topology (3-D with ``node_z``).

    Validates indices, drops isolated (degree-0) non-inlet nodes left by
    deletions, reindexes, re-derives ``node_role`` by degree and recomputes
    ``seg_length``/``seg_angle``. Connected fragments *without* an inlet are kept
    — their leaves couple to the Darcy bed, so they stay solvable.
    ``template`` (the tree being edited) may be ``None`` for a tree drawn from
    scratch; without ``inlet_nodes`` it is then fed at ``root_node`` (else node 0).
    """
    from ..core.geometry import Network, NodeRole

    nx = np.asarray(node_x, float)
    ny = np.asarray(node_y, float)
    ss = np.asarray(seg_start, int)
    se = np.asarray(seg_end, int)
    radius = np.maximum(np.asarray(seg_radius, float), 1e-3)
    n = len(nx)
    nz = None if node_z is None else np.asarray(node_z, float)
    if len(ny) != n or (nz is not None and len(nz) != n):
        raise HTTPException(422, "node_x, node_y (and node_z) must be the same length")
    if len(ss) != len(se) or len(radius) != len(ss):
        raise HTTPException(422, "seg_start, seg_end and seg_radius must align")
    if len(ss) and (ss.min() < 0 or se.min() < 0 or ss.max() >= n or se.max() >= n):
        raise HTTPException(422, "a segment references an out-of-range node")

    if inlet_nodes is not None:
        inl = np.array(sorted({int(i) for i in inlet_nodes if 0 <= int(i) < n}), int)
    elif template is None:                # a brand-new tree: fed at its root (or node 0)
        inl = np.array([root_node if root_node is not None and 0 <= root_node < n else 0], int)
    else:
        prev = template.inlet_nodes if template.inlet_nodes is not None else [template.root_node]
        inl = np.array(sorted({int(i) for i in prev if 0 <= int(i) < n}), int)

    deg = np.zeros(n, int)
    if len(ss):
        np.add.at(deg, ss, 1)
        np.add.at(deg, se, 1)
    keep = deg > 0
    keep[inl] = True                      # keep prescribed nodes even if isolated
    if not keep.any():
        raise HTTPException(422, "the edit leaves an empty network")

    old2new = np.full(n, -1, int)
    old2new[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    seg_ok = keep[ss] & keep[se]
    ss2, se2 = old2new[ss[seg_ok]], old2new[se[seg_ok]]
    r2 = radius[seg_ok]
    nx2, ny2 = nx[keep], ny[keep]
    nz2 = None if nz is None else nz[keep]
    inl2 = old2new[inl] if inl.size else np.array([0], int)

    root2 = 0
    if root_node is not None and 0 <= root_node < n and keep[root_node]:
        root2 = int(old2new[root_node])
    elif inl2.size:
        root2 = int(inl2[0])

    new_deg = np.zeros(len(nx2), int)
    if len(ss2):
        np.add.at(new_deg, ss2, 1)
        np.add.at(new_deg, se2, 1)
    role = np.where(new_deg >= 2, int(NodeRole.BIFURCATION), int(NodeRole.TERMINAL))
    role[inl2] = int(NodeRole.ROOT)
    return Network(
        node_x=nx2, node_y=ny2, node_role=role, seg_start=ss2, seg_end=se2,
        seg_length=segment_lengths(nx2, ny2, nz2, ss2, se2),
        seg_radius=r2, seg_angle=np.arctan2(ny2[se2] - ny2[ss2], nx2[se2] - nx2[ss2]),
        root_node=root2, inlet_nodes=inl2, node_z=nz2,
    )


def segment_lengths(nx, ny, nz, ss, se) -> np.ndarray:
    """Euclidean segment lengths in 2-D, or 3-D when ``nz`` is given."""
    d2 = (nx[se] - nx[ss]) ** 2 + (ny[se] - ny[ss]) ** 2
    if nz is not None:
        d2 = d2 + (nz[se] - nz[ss]) ** 2
    return np.sqrt(d2)


def tracer_aif(d: dict) -> np.ndarray:
    """The tracer's stored arterial input (old blobs: rebuild the bolus shape)."""
    from ..core import tkm

    if "aif" in d:
        return np.asarray(d["aif"], float)
    times = d["times"]
    dur = float(times[-1]) if len(times) else 1.0
    ca = tkm.gamma_variate_aif(times, t0=0.08 * dur, A=3.0, B=max(0.06 * dur, 1e-6))
    return ca / ca.max() if ca.max() > 0 else ca


def tracer_map(d: dict, field: str, method: str = "semi", oi: float = 0.1,
               circulant: bool = False) -> np.ndarray:
    """A voxel-wise kinetic map (2-D or 3-D) from a stored tracer: ``semi``
    (max-slope ``Fp`` / AUC ``vp``) or ``deconv`` (SVD ``Fp`` / ``CBV`` / ``MTT``)."""
    from ..core import tkm

    times, frames, mask, ca = d["times"], d["frames"], d["mask"], tracer_aif(d)
    if method == "deconv":
        return tkm.deconvolution_maps(frames, times, ca, mask=mask, oi=float(oi),
                                      circulant=bool(circulant))[field]
    return tkm.kinetic_maps(frames, times, ca, mask=mask)[field]
