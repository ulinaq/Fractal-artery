"""Flow-video rendering: contrast-bolus frames → GIF / MP4 (2-D heatmaps, or a 3-D
maximum-intensity projection with the vessel trees, turning round the organ).

A service-layer module (NumPy + matplotlib colormaps + imageio, all imported lazily),
used by the web API's ``tracer.gif`` / ``tracer.mp4`` and the bundle, and by the MCP
``export_tissue_fields(video=True)`` tool — so neither depends on the other.
``project_3d`` is the same camera as ``frontend/viewer3d.js``'s ``project``: keep them
in step.
"""

from __future__ import annotations

import numpy as np


def segment_lines_3d(net):
    """Yield each segment as ``(x0, y0, z0, x1, y1, z1, radius)`` (3-D networks)."""
    for s in range(net.n_segments):
        a, b = int(net.seg_start[s]), int(net.seg_end[s])
        yield (float(net.node_x[a]), float(net.node_y[a]), float(net.node_z[a]),
               float(net.node_x[b]), float(net.node_y[b]), float(net.node_z[b]),
               float(net.seg_radius[s]))


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
