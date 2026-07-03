"""Paper B/C: tracer transport montage (inflow -> perfusion -> washout).

Solves the hybrid flow on the synthetic tree, injects a gamma-variate bolus at
the arterial roots, transports it through the capillary bed, and shows snapshots
over time -- the qualitative behaviour of a DCE-MRI acquisition.
Run with ``uv run python -m perfusion.examples.tracer_montage``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import (
    Grid,
    Parameters,
    arterial_tree,
    geometry,
    run_hybrid,
    tkm,
    transport,
    venous_tree,
    viz,
)

OUT = Path(__file__).resolve().parents[3] / "outputs"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    viz.apply_style()
    prm = Parameters(seed=0)

    # Hybrid solve; run_hybrid exposes the face fluxes and terminal sources that
    # the tracer transport needs, so no pipeline internals are reused here.
    xmin, xmax, ymin, ymax = geometry.bounding_box(
        arterial_tree(prm), venous_tree(prm), margin=0.06)
    width, height = xmax - xmin, ymax - ymin
    nx = 150
    grid = Grid(nx=nx, ny=round(nx * height / width), lx=width, ly=height)
    res = run_hybrid(prm, grid, domain_limit=(xmin, xmax, ymin, ymax), eps=4.0)

    # Bolus injection and transport (implicit, unconditionally stable).
    dt = 0.05
    t = np.arange(0.0, 40.0, dt)
    aif = tkm.gamma_variate_aif(t, t0=2.0, A=3.0, B=1.0)
    frames = transport.transport_forward(
        res.grid, res.fx, res.fy, res.sources, res.solution.terminal_flux,
        aif, dt=dt, theta=1.0)

    snapshots = [3, 5, 7, 10, 14, 18, 24, 32, 40]
    fig = viz.plot_tracer_montage(frames, snapshots, dt, extent=res.domain_limit,
                                  ncols=3)
    fig.suptitle("Tracer transport: inflow → perfusion → washout (Paper B/C)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tracer_montage.png")
    print(f"peak tissue concentration: {frames.max():.4g}")
    print(f"saved {OUT / 'tracer_montage.png'}")


if __name__ == "__main__":
    main()
