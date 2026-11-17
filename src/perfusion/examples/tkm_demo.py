"""Paper C: synthesise a tracer curve and compare tracer-kinetic models.

Injects a gamma-variate AIF, builds a synthetic tissue curve, and fits
MS / 1CM / 2CUM / 2CXM. Prints the estimates and saves a figure comparing the
fits and the recovered plasma flow.
Run with ``uv run python -m perfusion.examples.tkm_demo``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perfusion import tkm, viz

OUT = Path(__file__).resolve().parents[3] / "outputs"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz.apply_style()

    t = np.arange(0.0, 120.0, 0.5)
    ca = tkm.gamma_variate_aif(t, t0=5.0, A=3.0, B=1.5)

    # Ground-truth 1CM tissue curve (a network-flow model produces such curves).
    Fp_true, vp_true = 40.0, 8.0
    tissue = tkm.one_compartment(t, ca, Fp_true, vp_true)

    print(f"ground truth: Fp={Fp_true}, vp={vp_true}")
    fits, estimates = {}, {}
    for model in ("MS", "1CM", "2CUM", "2CXM"):
        r = tkm.fit(t, ca, tissue, model)
        fits[model] = r.curve
        estimates[model] = r.Fp
        vp = "n/a" if r.vp != r.vp else f"{r.vp:.3g}"
        print(f"  {model:5s}: Fp={r.Fp:8.3f}   vp={vp}")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.8))
    viz.plot_tkm(a0, t, ca, tissue, fits)
    a0.set_title("Tracer-kinetic model fits")
    viz.plot_parameter_bars(a1, estimates, truth=Fp_true, param="Fp")
    a1.set_title("Recovered plasma flow")
    fig.suptitle("Tracer-kinetic model comparison (Paper C)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tkm_fits.png")
    print(f"saved {OUT / 'tkm_fits.png'}")


if __name__ == "__main__":
    main()
