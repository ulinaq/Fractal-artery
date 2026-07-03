"""Paper D: hybrid flow on a synthetic fractal tree + a flux/TOF optimisation step.

Run with ``uv run perfusion-tree`` or ``uv run python -m perfusion.examples.tree_optimize``.
Saves a figure to ``outputs/tree_hybrid.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import Grid, Parameters, arterial_tree, geometry, run_hybrid, venous_tree, viz

OUT = Path(__file__).resolve().parents[3] / "outputs"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz.apply_style()
    prm = Parameters()
    # Size the capillary grid to the trees so the field and overlay align exactly.
    xmin, xmax, ymin, ymax = geometry.bounding_box(
        arterial_tree(prm), venous_tree(prm), margin=0.06)
    width, height = xmax - xmin, ymax - ymin
    nx = 150
    grid = Grid(nx=nx, ny=round(nx * height / width), lx=width, ly=height)
    res = run_hybrid(prm, grid, domain_limit=(xmin, xmax, ymin, ymax), eps=4.0)
    ext = res.domain_limit

    print(f"arterial nodes/segments: {res.art.n_nodes}/{res.art.n_segments}")
    print(f"venous   nodes/segments: {res.ven.n_nodes}/{res.ven.n_segments}")
    print(f"root inflow  q_a[0] = {res.solution.q_a[0]:.4g}")
    print(f"root outflow q_v[0] = {res.solution.q_v[0]:.4g}")
    v = res.flux
    print(f"flux range = {v.max() - v.min():.4g}, std = {v.std():.4g}")
    tof = res.tof[np.isfinite(res.tof) & (res.tof > 0)]
    print(f"time-of-flight std = {tof.std():.4g}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))

    # (a) the two trees over the coupled pressure field (aligned via extent)
    viz.plot_field(axes[0], res.pressure, extent=ext, title="Capillary pressure",
                   label="pressure [Pa]")
    viz.plot_network(axes[0], res.art, color=viz.ARTERY)
    viz.plot_network(axes[0], res.ven, color=viz.VEIN)

    # (b) arterial network coloured by |flow|
    q = np.abs(res.solution.q_a)
    axes[1].set_title("Arterial flow")
    viz.plot_network(axes[1], res.art, values=q, cmap="viridis", label="|q| [mL/s]")
    axes[1].set_aspect("equal")

    # (c) Darcy flux magnitude, (d) time-of-flight
    viz.plot_field(axes[2], res.flux, extent=ext, cmap=viz.CMAP_FLUX,
                   title="Darcy flux magnitude", label="|v| [mm/s]")
    viz.plot_field(axes[3], res.tof, extent=ext, cmap=viz.CMAP_TOF,
                   title="Time of flight", label="T [s]", log=True)

    fig.suptitle("Hybrid multiscale flow on a synthetic fractal tree (Paper D)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tree_hybrid.png")
    print(f"saved {OUT / 'tree_hybrid.png'}")


if __name__ == "__main__":
    main()
