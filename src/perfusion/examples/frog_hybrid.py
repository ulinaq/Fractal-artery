"""Paper A: hybrid model on the frog-tongue organ, using the *segmented* vessels.

Segments the arterial and venous networks from the bundled frog images
(``frog_art_low.png``, ``frog_vein.png``) and couples them through a Darcy
capillary bed on the tongue mask -- the full frog-tongue system. Falls back to
synthetic fractal trees if the image extra is unavailable.

Run with ``uv run perfusion-frog``. Needs the image extra:
``uv sync --extra image``. Saves ``outputs/frog_hybrid.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import Grid, Parameters, run_hybrid, viz

OUT = Path(__file__).resolve().parents[3] / "outputs"


def _segmented_system():
    """Return (prm, grid, domain_limit, artery, vein) from the segmented images."""
    from perfusion import segmentation as seg

    system = seg.load_frog_system(min_size=30)
    print(f"segmented arteries: {system.artery.n_segments} segments, "
          f"{system.artery.terminal_nodes.size} terminals")
    print(f"segmented veins:    {system.vein.n_segments} segments, "
          f"{system.vein.terminal_nodes.size} terminals")
    return Parameters(), system.grid, system.domain_limit, system.artery, system.vein


def _synthetic_system():
    """Fallback: synthetic trees on a rectangular domain (no image extra)."""
    grid = Grid(nx=140, ny=160, lx=140, ly=160)
    return Parameters(), grid, (-70, 70, 0, 160), None, None


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        prm, grid, domain_limit, art, ven = _segmented_system()
        subtitle = "segmented vasculature"
    except ImportError:
        print("frog images need the image extra: uv sync --extra image")
        print("falling back to synthetic trees on a rectangular domain")
        prm, grid, domain_limit, art, ven = _synthetic_system()
        subtitle = "synthetic trees (install the image extra for the real data)"

    viz.apply_style()
    res = run_hybrid(prm, grid, domain_limit=domain_limit, eps=4.0, art=art, ven=ven)
    ext = res.domain_limit
    print(f"pressure range (organ): "
          f"{res.pressure[grid.mask].min():.4g} .. {res.pressure[grid.mask].max():.4g}")

    pressure = np.where(grid.mask, res.pressure, np.nan)  # grey outside the organ

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    viz.plot_field(axes[0], pressure, extent=ext, title="Capillary pressure",
                   label="pressure [Pa]")
    viz.plot_network(axes[0], res.art, color=viz.ARTERY, lw=0.5)
    viz.plot_network(axes[0], res.ven, color=viz.VEIN, lw=0.5)
    viz.plot_field(axes[1], res.tof, extent=ext, cmap=viz.CMAP_TOF,
                   title="Time of flight", label="T [s]", log=True)
    for ax in axes:                                 # frame to the organ
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.suptitle(f"Hybrid model on the frog-tongue organ ({subtitle}) — Paper A",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "frog_hybrid.png")
    print(f"saved {OUT / 'frog_hybrid.png'}")


if __name__ == "__main__":
    main()
