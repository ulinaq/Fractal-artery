"""Plotting helpers for networks, capillary-bed fields, tracer curves.

Design choices follow a few visualization rules that hold regardless of medium:

- **Magnitude fields** (pressure, flux, time-of-flight) use *perceptually
  uniform, colorblind-safe sequential* colormaps (viridis / magma / cividis) --
  never a rainbow. Inactive (masked) cells render as a neutral grey.
- **The vascular network overlay is drawn in the same coordinates as the field**
  (pass ``extent=result.domain_limit``) so the two align.
- **Categorical series** (the tracer-kinetic models) use a fixed-order
  Okabe-Ito palette (validated colorblind-safe) with a legend -- identity is
  never colour-alone.
- Axes are recessive; every field carries a labelled colorbar.
"""

from __future__ import annotations

import numpy as np

from .core.geometry import Network

# Perceptually-uniform sequential maps, one per physical field.
CMAP_PRESSURE = "viridis"
CMAP_FLUX = "magma"
CMAP_TOF = "cividis"
BAD_COLOR = "#eceff1"          # inactive / NaN cells

# Vascular overlay tones.
ARTERY = "#c62828"
VEIN = "#1565c0"

# Fixed-order, colorblind-safe categorical palette (Okabe-Ito) for TKM series.
SERIES = {
    "1CM": "#0072B2",   # blue
    "MS": "#E69F00",    # orange
    "2CUM": "#009E73",  # green
    "2CXM": "#D55E00",  # vermillion
}
GROUND_TRUTH = "#333333"


def apply_style() -> None:
    """Apply a clean, consistent matplotlib style (call once per figure script)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9e9e9e",
        "axes.linewidth": 0.8,
        "xtick.color": "#616161",
        "ytick.color": "#616161",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
    })


def plot_field(ax, field: np.ndarray, extent=None, cmap: str = CMAP_PRESSURE,
               title: str | None = None, label: str | None = None,
               log: bool = False, colorbar: bool = True):
    """Draw a ``(nx, ny)`` capillary-bed field as an image with a colorbar.

    NaN/masked cells (e.g. outside the organ) are shown in a neutral grey.
    ``extent = (xmin, xmax, ymin, ymax)`` places the field in model coordinates
    so a network overlay lines up.
    """
    import matplotlib as mpl

    data = np.ma.masked_invalid(field.T)          # (ny, nx) for imshow row=y
    if log:
        data = np.ma.masked_less_equal(data, 0)
        norm = mpl.colors.LogNorm(vmin=data.min(), vmax=data.max())
    else:
        norm = None
    cm = mpl.colormaps[cmap].with_extremes(bad=BAD_COLOR)
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cm, norm=norm,
                   aspect="equal", interpolation="nearest")
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if label:
            cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return im


def plot_network(ax, net: Network, color: str = ARTERY, lw: float = 0.4,
                 values: np.ndarray | None = None, cmap: str = "magma",
                 label: str | None = None):
    """Draw a vascular tree; line width scales with radius.

    If ``values`` (one scalar per segment, e.g. |flow|) is given, segments are
    coloured by it with a sequential colormap and a colorbar is added.
    """
    from matplotlib.collections import LineCollection

    segs = np.stack([
        np.column_stack([net.node_x[net.seg_start], net.node_y[net.seg_start]]),
        np.column_stack([net.node_x[net.seg_end], net.node_y[net.seg_end]]),
    ], axis=1)
    widths = lw + 2.6 * net.seg_radius / net.seg_radius.max()

    lc = LineCollection(segs, linewidths=widths, capstyle="round")
    if values is not None:
        lc.set_array(np.asarray(values))
        lc.set_cmap(cmap)
    else:
        lc.set_color(color)
    ax.add_collection(lc)
    ax.autoscale_view()
    if values is not None:
        cb = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
        if label:
            cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return lc


def plot_tkm(ax, t: np.ndarray, aif: np.ndarray, tissue: np.ndarray,
             fits: dict, truth: str | None = None):
    """Plot the tissue curve and tracer-kinetic-model fits (fixed-order palette).

    ``fits`` maps model name -> fitted curve (skip ``None``). A legend is always
    shown so series identity is not colour-alone.
    """
    ax.plot(t, aif, color="#9e9e9e", lw=1.2, ls=":", label="AIF")
    ax.plot(t, tissue, color=GROUND_TRUTH, lw=2.4, label="tissue (truth)", zorder=5)
    for name in ("1CM", "MS", "2CUM", "2CXM"):        # fixed order
        curve = fits.get(name)
        if curve is None:
            continue
        ax.plot(t, curve, color=SERIES[name], lw=1.6, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("concentration [mM]")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, alpha=0.5)


def plot_parameter_bars(ax, estimates: dict, truth: float, param: str = "Fp"):
    """Grouped bars of estimated plasma flow per model vs. the ground truth."""
    names = [n for n in ("MS", "1CM", "2CUM", "2CXM") if n in estimates]
    vals = [estimates[n] for n in names]
    colors = [SERIES[n] for n in names]
    ax.bar(names, vals, color=colors, width=0.65)
    ax.axhline(truth, color=GROUND_TRUTH, ls="--", lw=1.4,
               label=f"truth = {truth:g}")
    for i, v in enumerate(vals):                       # direct value labels
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"estimated {param}")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)


def plot_tracer_montage(frames: np.ndarray, times: np.ndarray, dt: float,
                        extent=None, ncols: int = 5, cmap: str = "inferno"):
    """Montage of tracer-concentration snapshots (inflow -> perfusion -> washout).

    ``frames`` is ``(n_steps, nx, ny)``; ``times`` are the (seconds) snapshots to
    show. Returns the created figure.
    """
    import matplotlib.pyplot as plt

    idx = np.clip((np.asarray(times) / dt).astype(int), 0, len(frames) - 1)
    vmax = np.percentile(frames[idx], 99.5)
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, k, tt in zip(np.ravel(axes), idx, np.asarray(times), strict=False):
        ax.imshow(frames[k].T, origin="lower", extent=extent, cmap=cmap,
                  vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.set_title(f"t = {tt:g} s", fontsize=9)
    fig.tight_layout()
    return fig
