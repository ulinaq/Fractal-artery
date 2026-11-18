"""Tests for data loaders and the bundled synthetic dataset."""

import numpy as np
import pytest

from perfusion import io, tkm


def test_load_synthetic_perfusion():
    data = io.load_synthetic_perfusion()
    assert set(data) >= {"t", "aif", "ground_truth", "tissue_curves"}
    assert data["tissue_curves"].shape[0] == data["ground_truth"].shape[0]
    assert data["aif"].shape == data["t"].shape


def test_synthetic_dataset_is_fittable():
    # The bundled curves should recover their ground-truth (Fp, vp) via 1CM.
    data = io.load_synthetic_perfusion()
    t, aif = data["t"], data["aif"]
    for (Fp, vp), curve in zip(data["ground_truth"], data["tissue_curves"], strict=True):
        r = tkm.fit(t, aif, curve, "1CM")
        assert r.Fp == pytest.approx(Fp, rel=2e-2)
        assert r.vp == pytest.approx(vp, rel=2e-2)


def test_load_frog_trees():
    trees = io.load_frog_trees()
    assert isinstance(trees, dict)
    assert any("tree" in k for k in trees)


def test_frog_domain_mask():
    mask = io.frog_domain_mask()
    assert mask.dtype == bool
    assert mask.ndim == 2
    assert mask.any()


def test_load_mat_roundtrip(tmp_path):
    import scipy.io as sio

    path = tmp_path / "sample.mat"
    sio.savemat(path, {"vec": np.arange(5.0), "scalar": 3.0})
    loaded = io.load_mat(path)
    assert np.allclose(loaded["vec"], np.arange(5.0))


def test_results_summary_is_json_serialisable(hybrid_result):
    import json

    summary = io.results_summary(hybrid_result)
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)
    assert summary["arterial_inflow"] > 0
    json.dumps(summary)          # must contain only plain floats/ints


def test_save_results_csv(tmp_path, hybrid_result):
    path = io.save_results_csv(hybrid_result, tmp_path / "fields.csv")
    lines = path.read_text().splitlines()
    assert lines[0] == "x,y,pressure,flux,tof"
    assert len(lines) - 1 == int(hybrid_result.grid.mask.sum())   # one row per active cell
    assert len(lines[1].split(",")) == 5


def test_save_results_vtk_is_valid_imagedata(tmp_path, hybrid_result):
    import xml.etree.ElementTree as ET

    path = io.save_results_vtk(hybrid_result, tmp_path / "fields.vti")
    root = ET.fromstring(path.read_text())          # must be well-formed XML
    assert root.attrib["type"] == "ImageData"
    names = {da.attrib["Name"] for da in root.iter("DataArray")}
    assert {"pressure", "flux", "tof", "active"} <= names
    nx, ny = hybrid_result.pressure.shape
    pressure = next(da for da in root.iter("DataArray") if da.attrib["Name"] == "pressure")
    assert len(pressure.text.split()) == nx * ny    # one value per cell


def test_fields_export_in_3d(tmp_path):
    """3-D fields: CSV rows are x-fastest voxel centres + values for active voxels;
    the VTI is a voxel volume with the right origin / spacing."""
    import xml.etree.ElementTree as ET

    import numpy as np

    shape, dl = (4, 3, 2), (0.0, 8.0, -3.0, 3.0, 10.0, 14.0)
    p = np.arange(24, dtype=float).reshape(shape, order="F")    # value == flat index
    mask = np.ones(shape, bool)
    mask[0, 0, 0] = False
    path = io.fields_to_csv(tmp_path / "f.csv", dl, pressure=p, flux=p, tof=p, mask=mask)
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    assert path.read_text().splitlines()[0] == "x,y,z,pressure,flux,tof"
    assert len(rows) == 23
    assert rows[0].tolist() == [3.0, -2.0, 11.0, 1.0, 1.0, 1.0]   # voxel (1,0,0)
    assert rows[-1].tolist() == [7.0, 2.0, 13.0, 23.0, 23.0, 23.0]

    vti = io.fields_to_vtk(tmp_path / "f.vti", dl, pressure=p, flux=p, tof=p, mask=mask)
    img = ET.fromstring(vti.read_text()).find("ImageData")
    assert img.attrib["WholeExtent"] == "0 4 0 3 0 2"
    assert img.attrib["Origin"] == "0 -3 10"
    assert img.attrib["Spacing"] == "2 2 2"
    with pytest.raises(ValueError, match="needs 6 domain_limit"):
        io.fields_to_csv(tmp_path / "x.csv", dl[:4], pressure=p, flux=p, tof=p)


def test_save_results_npz(tmp_path, hybrid_result):
    path = io.save_results(hybrid_result, tmp_path / "result.npz")
    assert path.exists()
    with np.load(path) as data:
        assert {"pressure", "flux", "tof", "q_a", "q_v", "terminal_flux",
                "domain_limit"} <= set(data.files)
        assert data["pressure"].shape == hybrid_result.pressure.shape


def test_network_to_vtp_roundtrips(tmp_path):
    import xml.etree.ElementTree as ET

    from perfusion import Parameters, arterial_tree
    from perfusion.io import network_to_vtp

    net = arterial_tree(Parameters(r_min_frac=0.5, seed=0))
    out = network_to_vtp(net, tmp_path / "tree.vtp", flow=None)
    assert out.exists()
    tree = ET.parse(out)                       # valid XML
    piece = tree.find(".//Piece")
    assert int(piece.attrib["NumberOfPoints"]) == net.n_nodes
    assert int(piece.attrib["NumberOfLines"]) == net.n_segments
    names = {d.attrib.get("Name") for d in tree.iter("DataArray")}
    assert {"connectivity", "offsets", "radius", "order"} <= names
