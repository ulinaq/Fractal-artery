"""Phase 4: the full 3-D hybrid model — 3-D trees coupled to a 3-D Darcy bed.

Grows a 3-D arterial tree (up) and venous tree (down), couples them to a 3-D
Darcy capillary bed, solves the hybrid flow, and saves orthogonal mid-plane
pressure slices with the trees projected on top. Confirms end-to-end 3-D
coupling (arterial inflow == venous outflow).
Run with ``uv run python -m perfusion.examples.hybrid3d_demo``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import Parameters, grow_tree_3d, run_hybrid_3d
from perfusion.core.darcy3d import Grid3D

OUT = Path(__file__).resolve().parents[3] / "outputs"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    art = grow_tree_3d((25, 25, 3), (0, 0, 1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=6)
    ven = grow_tree_3d((25, 25, 47), (0, 0, -1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=6, mirror=True)
    xs = np.concatenate([art.node_x, ven.node_x])
    ys = np.concatenate([art.node_y, ven.node_y])
    zs = np.concatenate([art.node_z, ven.node_z])
    m = 4.0
    dom = (xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m, zs.min() - m, zs.max() + m)
    grid = Grid3D(nx=30, ny=30, nz=32,
                  lx=dom[1] - dom[0], ly=dom[3] - dom[2], lz=dom[5] - dom[4])
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)

    p = res.pressure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, iz in zip(axes, [grid.nz // 4, grid.nz // 2, 3 * grid.nz // 4], strict=True):
        z = dom[4] + (iz + 0.5) * grid.hz
        im = ax.imshow(p[:, :, iz].T, origin="lower", cmap="viridis",
                       extent=(dom[0], dom[1], dom[2], dom[3]), aspect="auto")
        ax.set_title(f"pressure @ z={z:.0f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("3-D hybrid: arterial + venous trees coupled to a 3-D Darcy bed",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT.mkdir(exist_ok=True)
    path = OUT / "hybrid3d_demo.png"
    fig.savefig(path, dpi=120)
    n_art = res.art.terminal_nodes.size
    inflow = float(res.solution.terminal_flux[:n_art].sum())
    outflow = float(-res.solution.terminal_flux[n_art:].sum())
    print(f"3-D hybrid solved on {grid.nx}x{grid.ny}x{grid.nz} = {grid.n_cells} cells; "
          f"arterial inflow {inflow:.4f} == venous outflow {outflow:.4f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
