"""Data loaders (.mat, images, bundled datasets) and result exporters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as sio

if TYPE_CHECKING:
    from .core.pipeline import HybridResult

# Repo layout: this file is src/perfusion/io.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LEGACY_DIR = REPO_ROOT / "legacy"


def load_mat(path: str | Path) -> dict:
    """Load a MATLAB .mat file (v7 and earlier) into a dict of arrays."""
    return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def frog_domain_mask(path: str | Path | None = None) -> np.ndarray:
    """Load the frog-tongue domain as a boolean mask (nx, ny).

    Requires the optional ``image`` extra (``uv sync --extra image``). Falls back
    to the copy in ``data/`` if no path is given.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the 'image' extra: uv sync --extra image") from exc
    if path is None:
        path = DATA_DIR / "frog_domain_low.png"
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img > (img.max() / 2)


def load_frog_trees(path: str | Path | None = None) -> dict:
    """Load the segmented frog arterial/venous trees (``frog_trees.mat``)."""
    if path is None:
        path = DATA_DIR / "frog_trees.mat"
    return load_mat(path)


def load_synthetic_perfusion(path: str | Path | None = None) -> dict:
    """Load the bundled synthetic DCE-MRI dataset (``data/synthetic_perfusion.npz``).

    Returns a dict with ``t`` (time grid), ``aif`` (arterial input function),
    ``ground_truth`` (``(n, 2)`` array of ``[Fp, vp]``) and ``tissue_curves``
    (``(n, len(t))``). Regenerate with ``scripts/generate_data.py``.
    """
    if path is None:
        path = DATA_DIR / "synthetic_perfusion.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# Result export                                                               #
# --------------------------------------------------------------------------- #

def results_summary(result: HybridResult) -> dict:
    """JSON-serialisable scalar summary of a hybrid simulation result."""
    art, ven, sol = result.art, result.ven, result.solution
    n_art = art.terminal_nodes.size
    tof = result.tof[np.isfinite(result.tof)]
    flux = result.flux[np.isfinite(result.flux)]
    return {
        "arterial_nodes": int(art.n_nodes),
        "arterial_segments": int(art.n_segments),
        "venous_nodes": int(ven.n_nodes),
        "venous_segments": int(ven.n_segments),
        "arterial_inflow": float(sol.terminal_flux[:n_art].sum()),
        "venous_outflow": float(-sol.terminal_flux[n_art:].sum()),
        "pressure_min": float(np.nanmin(result.pressure)),
        "pressure_max": float(np.nanmax(result.pressure)),
        "flux_mean": float(flux.mean()) if flux.size else 0.0,
        "flux_std": float(flux.std()) if flux.size else 0.0,
        "tof_mean": float(tof.mean()) if tof.size else 0.0,
        "tof_std": float(tof.std()) if tof.size else 0.0,
    }


def save_results(result: HybridResult, path: str | Path) -> Path:
    """Save the fields and network flows of a hybrid result to a ``.npz`` file."""
    path = Path(path)
    np.savez(
        path,
        pressure=result.pressure, flux=result.flux, tof=result.tof,
        domain_limit=np.asarray(result.domain_limit),
        q_a=result.solution.q_a, q_v=result.solution.q_v,
        terminal_flux=result.solution.terminal_flux,
    )
    return path if path.suffix else path.with_suffix(".npz")
