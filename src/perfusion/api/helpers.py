"""Compute helpers shared by the route modules (no routing, no state).

Kept out of the routers so the request handlers stay thin and these stay unit-
testable: the frog/demo 3-D scene builders, the edited-network rebuild, and the
tracer GIF / MP4 renderers.
"""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException

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


def segment_lines_3d(net):
    """Yield each segment as ``(x0, y0, z0, x1, y1, z1, radius)`` (3-D networks)."""
    for s in range(net.n_segments):
        a, b = int(net.seg_start[s]), int(net.seg_end[s])
        yield (float(net.node_x[a]), float(net.node_y[a]), float(net.node_z[a]),
               float(net.node_x[b]), float(net.node_y[b]), float(net.node_z[b]),
               float(net.seg_radius[s]))


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


def _tracer_images(tracer: dict, scene: dict | None = None) -> list:
    """Tracer frames colour-mapped (viridis) to upright RGB images (row = y down).
    A 3-D tracer (3-D ``mask``) is rendered with :func:`tracer_images_3d` from the
    ``scene`` (``domain`` + vessel ``segments``)."""
    from matplotlib import colormaps

    if np.ndim(tracer["mask"]) == 3:
        if scene is None:
            raise ValueError("a 3-D tracer needs a scene (domain + segments) to render")
        return tracer_images_3d(tracer, **scene)
    frames, mask, vmax = tracer["frames"], tracer["mask"], float(tracer["vmax"]) or 1.0
    cmap = colormaps["viridis"]
    images = []
    for f in frames:
        rgb = (cmap(np.clip(f / vmax, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
        rgb[~mask] = 245                        # inactive tissue -> light grey
        images.append(np.transpose(rgb, (1, 0, 2))[::-1])   # (nx,ny)->image (row=y down)
    return images


# camera of the 3-D viewer (frontend/viewer3d.js): yaw about z, then pitch about x;
# screen x ← x', screen up ← z', depth = y'; the scene spans ±0.62 of the frame.
_VIEW_YAW, _VIEW_PITCH, _VIEW_HALF = 0.7, 0.45, 0.62


def project_3d(xyz, center, span, yaw, pitch, size):
    """Orthographic projection of ``(n, 3)`` points into a ``size``-px square frame,
    matching the 3-D viewer's camera. Returns ``(col, row, depth)`` arrays."""
    d = (np.asarray(xyz, float) - center) / span
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x = d[:, 0] * cy - d[:, 1] * sy
    y = d[:, 0] * sy + d[:, 1] * cy
    depth, up = y * cp - d[:, 2] * sp, y * sp + d[:, 2] * cp
    scale = size / (2 * _VIEW_HALF)
    return size / 2 + x * scale, size / 2 - up * scale, depth


def tracer_images_3d(tracer: dict, domain, segments, size: int = 480,
                     spin: float = 2 * np.pi, yaw: float = _VIEW_YAW,
                     pitch: float = _VIEW_PITCH) -> list:
    """Render a 3-D tracer as RGB frames, viewed like the 3-D viewer.

    Each frame is a **maximum-intensity projection** of the voxel concentration
    (viridis, the usual way to show a 3-D contrast bolus) over the organ's silhouette,
    with the arterial (red) / venous (blue) segments ``[x0,y0,z0,x1,y1,z1,r,kind]``
    drawn on top and a time bar along the bottom. The camera turns by ``spin`` radians
    over the clip (0 = fixed view).
    """
    from matplotlib import colormaps

    frames, mask = np.asarray(tracer["frames"]), np.asarray(tracer["mask"], bool)
    vmax = float(tracer["vmax"]) or 1.0
    lo, hi = np.asarray(domain[0::2], float), np.asarray(domain[1::2], float)
    center, span = (lo + hi) / 2, float((hi - lo).max())
    h = (hi - lo) / np.array(mask.shape)
    idx = np.argwhere(mask)
    xyz = lo + (idx + 0.5) * h
    vals = frames[:, idx[:, 0], idx[:, 1], idx[:, 2]]                   # (n_t, n_vox)
    seg = np.asarray(segments, float).reshape(-1, 8)
    n_t = len(frames)
    # each voxel splats to a square about its projected size
    rad = max(1, int(np.ceil(0.6 * h.max() / span * size / (2 * _VIEW_HALF))))
    offs = [(a, b) for a in range(-rad, rad + 1) for b in range(-rad, rad + 1)]
    cmap = colormaps["viridis"]
    bg, tissue = np.array([11, 16, 32], float), np.array([30, 41, 59], float)
    vessel_rgb = np.array([[255, 107, 107], [91, 141, 255]], float)
    images = []
    for i in range(n_t):
        ang = yaw + (spin * i / n_t if n_t > 1 else 0.0)
        col, row, _ = project_3d(xyz, center, span, ang, pitch, size)
        c0, r0 = np.round(col).astype(int), np.round(row).astype(int)
        mip = np.zeros((size, size))
        sil = np.zeros((size, size), bool)
        for a, b in offs:
            r, c = r0 + a, c0 + b
            ok = (r >= 0) & (r < size) & (c >= 0) & (c < size)
            np.maximum.at(mip, (r[ok], c[ok]), np.nan_to_num(vals[i, ok]))
            sil[r[ok], c[ok]] = True
        img = np.broadcast_to(bg, (size, size, 3)).copy()
        img[sil] = tissue
        t = np.clip(mip / vmax, 0.0, 1.0)
        lit = sil & (t > 0.02)
        alpha = (0.35 + 0.65 * t[lit])[:, None]
        img[lit] = (1 - alpha) * img[lit] + alpha * cmap(t[lit])[:, :3] * 255
        if len(seg):                                   # vessels: sampled along each segment
            a_c, a_r, _ = project_3d(seg[:, 0:3], center, span, ang, pitch, size)
            b_c, b_r, _ = project_3d(seg[:, 3:6], center, span, ang, pitch, size)
            n = np.maximum(2, np.ceil(np.hypot(b_c - a_c, b_r - a_r)).astype(int) + 1)
            k = np.repeat(np.arange(len(seg)), n)
            f = np.concatenate([np.linspace(0, 1, m) for m in n])
            pc = np.round(a_c[k] + f * (b_c - a_c)[k]).astype(int)
            pr = np.round(a_r[k] + f * (b_r - a_r)[k]).astype(int)
            thick = seg[k, 6] > np.median(seg[:, 6])           # wider trunks
            for dr, dc, use in ((0, 0, None), (1, 0, thick), (0, 1, thick)):
                rr, cc = pr + dr, pc + dc
                ok = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
                if use is not None:
                    ok &= use
                kind = seg[k[ok], 7].astype(int).clip(0, 1)
                img[rr[ok], cc[ok]] = 0.3 * img[rr[ok], cc[ok]] + 0.7 * vessel_rgb[kind]
        bar = round(size * (i + 1) / n_t)                      # time bar
        img[-4:, :bar] = (255, 255, 255)
        images.append(img.astype(np.uint8))
    return images


def render_gif(tracer: dict, fps: int = 12, scene: dict | None = None) -> bytes:
    """Colour-map tracer frames (viridis) into an animated GIF (needs imageio).
    A 3-D tracer needs the ``scene`` (see :func:`tracer_images_3d`)."""
    import io as _io

    import imageio.v2 as imageio

    buf = _io.BytesIO()
    imageio.mimsave(buf, _tracer_images(tracer, scene), format="GIF", fps=fps, loop=0)
    return buf.getvalue()


def render_mp4(tracer: dict, fps: int = 12, min_size: int = 480,
               scene: dict | None = None) -> bytes:
    """Encode the tracer frames as an H.264 MP4 (needs the ``video`` extra:
    imageio + imageio-ffmpeg, which ships its own ffmpeg binary).

    Frames are upscaled by an integer factor (nearest, so cells stay crisp) to at
    least ``min_size`` px and padded to a multiple of 16 for yuv420p / players.
    """
    import os
    import tempfile

    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa: F401 — fail early (ImportError) without the extra

    images = _tracer_images(tracer, scene)
    h, w = images[0].shape[:2]
    k = max(1, -(-min_size // max(1, min(h, w))))       # ceil(min_size / short side)
    H, W = -(-h * k // 16) * 16, -(-w * k // 16) * 16

    def frame(img):
        big = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
        out = np.full((H, W, 3), 245, np.uint8)
        out[:big.shape[0], :big.shape[1]] = big
        return out

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        imageio.mimwrite(path, [frame(i) for i in images], fps=fps, codec="libx264",
                         pixelformat="yuv420p", macro_block_size=1,
                         ffmpeg_log_level="error")
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)
