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



def _npz_written(path) -> Path:
    """Where ``np.savez(path)`` actually wrote: it appends ``.npz`` to any path that
    does not already end in it (``run.v1`` → ``run.v1.npz``)."""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

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
        **_rheology_summary(getattr(result, "rheology", None)),
    }


def _rheology_summary(info: dict | None) -> dict:
    """Non-Newtonian solve diagnostics: the viscosity-factor range (Carreau or Casson)
    and, for Casson blood, how many vessels are plugged below yield."""
    if not info:
        return {}
    out = {"viscosity_factor_min": info["factor_min"],
           "viscosity_factor_max": info["factor_max"],
           "rheology_iterations": int(info["iterations"])}
    if "plugged" in info:
        out["plugged_vessels"] = int(info["plugged"])
    return out


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
    return _npz_written(path)


def _cell_centres(shape, domain_limit):
    """Column-major (x-fastest) cell-centre coordinates over the domain window —
    ``(x, y)`` for a 2-D ``shape`` with a 4-value ``domain_limit``, ``(x, y, z)`` for
    a 3-D one with 6 values."""
    lim = np.asarray(domain_limit, dtype=float)
    axes = [lo + (np.arange(n) + 0.5) * (hi - lo) / n
            for n, lo, hi in zip(shape, lim[0::2], lim[1::2], strict=True)]
    return np.meshgrid(*axes, indexing="ij")


def _check_frame(pressure, domain_limit) -> tuple[int, ...]:
    shape = np.shape(pressure)
    if len(shape) not in (2, 3) or len(domain_limit) != 2 * len(shape):
        raise ValueError(f"a {len(shape)}-D field needs {2 * len(shape)} domain_limit "
                         f"values, got {len(domain_limit)}")
    return shape


def fields_to_csv(path: str | Path, domain_limit, *, pressure, flux, tof,
                  mask=None) -> Path:
    """Write the fields as a long-format CSV: one row per (active) cell.

    Columns ``x,y,pressure,flux,tof`` (``x,y,z,…`` for a 3-D field) at cell centres in
    the ``domain_limit`` frame. With ``mask`` given, only active cells are written
    (organ-shaped). Rows run x-fastest (column-major), in 2-D and 3-D alike.
    """
    shape = _check_frame(pressure, domain_limit)
    coords = _cell_centres(shape, domain_limit)
    cols = [*coords, pressure, flux, tof]
    flat = np.column_stack([np.asarray(c, dtype=float).ravel(order="F") for c in cols])
    if mask is not None:
        flat = flat[np.asarray(mask, dtype=bool).ravel(order="F")]
    names = "xyz"[:len(shape)]
    path = Path(path)
    np.savetxt(path, flat, delimiter=",", header=",".join([*names, "pressure", "flux", "tof"]),
               comments="", fmt="%.6g")
    return path


def fields_to_vtk(path: str | Path, domain_limit, *, pressure, flux, tof,
                  mask=None) -> Path:
    """Write a VTK ImageData (``.vti``) file (ASCII) that opens in ParaView.

    Fields become cell data on a uniform grid over ``domain_limit`` — a one-cell-thick
    slab for a 2-D field, a voxel volume for a 3-D one; non-finite values are zeroed
    and an ``active`` array (from ``mask``) marks the organ so inactive cells can be
    thresholded out.
    """
    shape = _check_frame(pressure, domain_limit)
    lim = np.asarray(domain_limit, dtype=float)
    dims = [*shape, 1] if len(shape) == 2 else list(shape)
    origin = [*lim[0::2], 0.0][:3]
    spacing = [(hi - lo) / n for n, lo, hi in zip(shape, lim[0::2], lim[1::2], strict=True)]
    spacing = [*spacing, 1.0][:3]
    active = (np.ones(shape, bool) if mask is None else np.asarray(mask, bool))

    def _array(name, a, dtype="Float64"):
        vals = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        text = " ".join(f"{v:.6g}" for v in vals.ravel(order="F"))
        return (f'        <DataArray type="{dtype}" Name="{name}" format="ascii">'
                f"{text}</DataArray>")

    body = "\n".join([
        _array("pressure", pressure), _array("flux", flux), _array("tof", tof),
        _array("active", active.astype(int), "Int32"),
    ])
    extent = " ".join(f"0 {n}" for n in dims)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" '
        f'Origin="{" ".join(f"{v:.6g}" for v in origin)}" '
        f'Spacing="{" ".join(f"{v:.6g}" for v in spacing)}">\n'
        f'    <Piece Extent="{extent}">\n'
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


def network_to_vtp(net, path: str | Path, *, flow=None) -> Path:
    """Write a vascular network as VTK PolyData (``.vtp``) — points + line cells.

    Each node becomes a point (``z = node_z`` if present, else 0); each segment a
    2-point line cell carrying ``radius`` (and ``flow`` if given) as cell data, plus
    the Strahler ``order`` for colouring. Opens directly in ParaView as a set of
    tubes (use the *Tube* filter with the ``radius`` array).
    """
    node_x = np.asarray(net.node_x, dtype=float)
    node_y = np.asarray(net.node_y, dtype=float)
    node_z = np.asarray(getattr(net, "node_z", None)
                        if getattr(net, "node_z", None) is not None
                        else np.zeros_like(node_x), dtype=float)
    from .core.morphometry import strahler_orders
    order = strahler_orders(net)

    pts = " ".join(f"{x:.6g} {y:.6g} {z:.6g}"
                   for x, y, z in zip(node_x, node_y, node_z, strict=True))
    conn = " ".join(f"{int(a)} {int(b)}"
                    for a, b in zip(net.seg_start, net.seg_end, strict=True))
    offsets = " ".join(str(2 * (i + 1)) for i in range(net.n_segments))

    def _cell(name, a, dtype="Float64"):
        text = " ".join(f"{v:.6g}" for v in np.asarray(a, dtype=float))
        return (f'        <DataArray type="{dtype}" Name="{name}" format="ascii">'
                f"{text}</DataArray>")

    cell_arrays = [_cell("radius", net.seg_radius), _cell("order", order, "Int32")]
    if flow is not None:
        cell_arrays.append(_cell("flow", np.abs(np.asarray(flow, dtype=float))))
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
        "  <PolyData>\n"
        f'    <Piece NumberOfPoints="{net.n_nodes}" NumberOfLines="{net.n_segments}">\n'
        "      <Points>\n"
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        f"{pts}</DataArray>\n"
        "      </Points>\n"
        "      <Lines>\n"
        f'        <DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>\n'
        f'        <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        "      </Lines>\n"
        '      <CellData Scalars="radius">\n'
        f"{chr(10).join(cell_arrays)}\n"
        "      </CellData>\n"
        "    </Piece>\n"
        "  </PolyData>\n"
        "</VTKFile>\n"
    )
    path = Path(path)
    path.write_text(xml)
    return path                       # written exactly where asked


def save_network(net, path: str | Path) -> Path:
    """Serialise a :class:`~perfusion.core.geometry.Network` to a ``.npz`` file.

    ``inlet_nodes`` (the prescribed-pressure nodes — every inlet/outlet) and
    ``node_z`` (3-D trees) are stored when set, so they survive a round trip.
    """
    path = Path(path)
    extra = {}
    if net.inlet_nodes is not None:
        extra["inlet_nodes"] = np.asarray(net.inlet_nodes, dtype=int)
    if net.node_z is not None:
        extra["node_z"] = np.asarray(net.node_z, dtype=float)
    np.savez(
        path,
        node_x=net.node_x, node_y=net.node_y, node_role=net.node_role,
        seg_start=net.seg_start, seg_end=net.seg_end, seg_length=net.seg_length,
        seg_radius=net.seg_radius, seg_angle=net.seg_angle,
        root_node=np.asarray(net.root_node), **extra,
    )
    return _npz_written(path)


def load_network(path: str | Path):
    """Reconstruct a :class:`~perfusion.core.geometry.Network` saved by :func:`save_network`."""
    from .core.geometry import Network

    with np.load(path) as d:
        return Network(
            node_x=d["node_x"], node_y=d["node_y"], node_role=d["node_role"],
            seg_start=d["seg_start"], seg_end=d["seg_end"], seg_length=d["seg_length"],
            seg_radius=d["seg_radius"], seg_angle=d["seg_angle"],
            root_node=int(d["root_node"]),
            # older files predate these fields -> Network's defaults
            inlet_nodes=d["inlet_nodes"] if "inlet_nodes" in d.files else None,
            node_z=d["node_z"] if "node_z" in d.files else None,
        )
