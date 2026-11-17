"""Phase 4 foundation: a 3-D Darcy source -> sink solve, visualised as slices.

Builds a 3-D grid, injects flow at one corner region and extracts it at the
opposite one (a balanced dipole), solves the pressure field with the 3-D TPFA
operator, and saves three orthogonal mid-plane slices. Demonstrates that the
``perfusion.core.darcy3d`` foundation works end-to-end.
Run with ``uv run python -m perfusion.examples.darcy3d_demo``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion.core.darcy3d import Grid3D, point_source_3d, solve_pressure_3d

OUT = Path(__file__).resolve().parents[3] / "outputs"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = Grid3D(nx=28, ny=24, nz=20, lx=28, ly=24, lz=20)
    src = int(grid.sub2ind(4, 12, 10))            # injection
    snk = int(grid.sub2ind(23, 12, 10))           # extraction
    q = np.zeros(grid.n_cells)
    for cell, sign in ((src, +1.0), (snk, -1.0)):
        cells, weights = point_source_3d(grid, cell, eps=2.5)
        q[cells] += sign * weights
    p = solve_pressure_3d(grid, q)                # (nx, ny, nz)

    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    slices = [
        ("z mid-plane (x-y)", p[:, :, cz].T, (0, grid.lx, 0, grid.ly)),
        ("y mid-plane (x-z)", p[:, cy, :].T, (0, grid.lx, 0, grid.lz)),
        ("x mid-plane (y-z)", p[cx, :, :].T, (0, grid.ly, 0, grid.lz)),
    ]
    vlim = np.abs(p).max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl, extent) in zip(axes, slices, strict=True):
        im = ax.imshow(sl, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("3-D Darcy pressure: source (left) → sink (right)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT.mkdir(exist_ok=True)
    path = OUT / "darcy3d_demo.png"
    fig.savefig(path, dpi=120)
    print(f"3-D source->sink solved on {grid.nx}x{grid.ny}x{grid.nz} grid; "
          f"pressure range [{p.min():.3g}, {p.max():.3g}]")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
