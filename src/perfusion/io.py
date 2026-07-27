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
    from .core.perfusion_metrics import transit_metrics

    art, ven, sol = result.art, result.ven, result.solution
    n_art = art.terminal_nodes.size
    tof = result.tof[np.isfinite(result.tof)]
    flux = result.flux[np.isfinite(result.flux)]
    # flow-weighted transit-time distribution -> MTT, CTH (heterogeneity), RD
    tm = transit_metrics(result.tof, weights=result.flux)
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
        "mtt": tm.mtt,          # mean transit time (flow-weighted)
        "cth": tm.cth,          # capillary transit-time heterogeneity (std dev)
        "rd": tm.rd,            # relative dispersion = cth / mtt
    }


def save_results(result: HybridResult, path: str | Path) -> Path:
    """Save the fields and network flows of a hybrid result to a ``.npz`` file."""
    path = Path(path)
    np.savez(
        path,
        pressure=result.pressure, flux=result.flux, tof=result.tof,
        domain_limit=np.asarray(result.domain_limit),
        mask=np.asarray(result.grid.mask),
        q_a=result.solution.q_a, q_v=result.solution.q_v,
        terminal_flux=result.solution.terminal_flux,
    )
    return path if path.suffix else path.with_suffix(".npz")


def _cell_centres(nx: int, ny: int, domain_limit):
    """Column-major (x-fastest) cell-centre coordinates over the domain window."""
    xmin, xmax, ymin, ymax = domain_limit
    xc = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    yc = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    x, y = np.meshgrid(xc, yc, indexing="ij")
    return x, y


def fields_to_csv(path: str | Path, domain_limit, *, pressure, flux, tof,
                  mask=None) -> Path:
    """Write the fields as a long-format CSV: one row per (active) cell.

    Columns ``x,y,pressure,flux,tof`` at cell centres in the ``domain_limit``
    frame. With ``mask`` given, only active cells are written (organ-shaped).
    """
    nx, ny = pressure.shape
    x, y = _cell_centres(nx, ny, domain_limit)
    cols = [x, y, pressure, flux, tof]
    flat = np.column_stack([c.ravel(order="F") for c in cols])
    if mask is not None:
        flat = flat[np.asarray(mask, dtype=bool).ravel(order="F")]
    path = Path(path)
    np.savetxt(path, flat, delimiter=",", header="x,y,pressure,flux,tof",
               comments="", fmt="%.6g")
    return path


def fields_to_vtk(path: str | Path, domain_limit, *, pressure, flux, tof,
                  mask=None) -> Path:
    """Write a VTK ImageData (``.vti``) file (ASCII) that opens in ParaView.

    Fields become cell data on a uniform grid over ``domain_limit``; non-finite
    values are zeroed and an ``active`` array (from ``mask``) marks the organ so
    inactive cells can be thresholded out.
    """
    nx, ny = pressure.shape
    xmin, _, ymin, _ = domain_limit
    sx, sy = (domain_limit[1] - xmin) / nx, (domain_limit[3] - ymin) / ny
    active = (np.ones((nx, ny), bool) if mask is None else np.asarray(mask, bool))

    def _array(name, a, dtype="Float64"):
        vals = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        text = " ".join(f"{v:.6g}" for v in vals.ravel(order="F"))
        return (f'        <DataArray type="{dtype}" Name="{name}" format="ascii">'
                f"{text}</DataArray>")

    body = "\n".join([
        _array("pressure", pressure), _array("flux", flux), _array("tof", tof),
        _array("active", active.astype(int), "Int32"),
    ])
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 1" '
        f'Origin="{xmin} {ymin} 0" Spacing="{sx:.6g} {sy:.6g} 1">\n'
        f'    <Piece Extent="0 {nx} 0 {ny} 0 1">\n'
        '      <CellData Scalars="pressure">\n'
        f"{body}\n"
        "      </CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    path = Path(path)
    path.write_text(xml)
    return path


def save_results_csv(result: HybridResult, path: str | Path) -> Path:
    """Long-format CSV of a hybrid result's fields (see :func:`fields_to_csv`)."""
    return fields_to_csv(path, result.domain_limit, pressure=result.pressure,
                         flux=result.flux, tof=result.tof, mask=result.grid.mask)


def save_results_vtk(result: HybridResult, path: str | Path) -> Path:
    """VTK ImageData of a hybrid result's fields (see :func:`fields_to_vtk`)."""
    return fields_to_vtk(path, result.domain_limit, pressure=result.pressure,
                         flux=result.flux, tof=result.tof, mask=result.grid.mask)


def save_network(net, path: str | Path) -> Path:
    """Serialise a :class:`~perfusion.core.geometry.Network` to a ``.npz`` file."""
    path = Path(path)
    np.savez(
        path,
        node_x=net.node_x, node_y=net.node_y, node_role=net.node_role,
        seg_start=net.seg_start, seg_end=net.seg_end, seg_length=net.seg_length,
        seg_radius=net.seg_radius, seg_angle=net.seg_angle,
        root_node=np.asarray(net.root_node),
    )
    return path if path.suffix else path.with_suffix(".npz")


def load_network(path: str | Path):
    """Reconstruct a :class:`~perfusion.core.geometry.Network` saved by :func:`save_network`."""
    from .core.geometry import Network

    with np.load(path) as d:
        return Network(
            node_x=d["node_x"], node_y=d["node_y"], node_role=d["node_role"],
            seg_start=d["seg_start"], seg_end=d["seg_end"], seg_length=d["seg_length"],
            seg_radius=d["seg_radius"], seg_angle=d["seg_angle"],
            root_node=int(d["root_node"]),
        )
