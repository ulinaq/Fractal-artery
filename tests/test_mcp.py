"""Tests for the MCP server that exposes the perfusion engine.

The tool *logic* (``simulate`` / ``model_info``) needs no ``mcp`` package; the
server construction does (``pytest.importorskip``).
"""

import numpy as np
import pytest

from perfusion import mcp_server


def test_simulate_returns_a_conserving_summary():
    summary = mcp_server.simulate(nx=50, ny=60, r_min_frac=0.5)   # shallow -> fast
    assert summary["arterial_inflow"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)
    assert tuple(summary["grid"].values()) == (50, 60)


def test_simulate_clamps_resolution():
    # absurd sizes are clamped into a safe range rather than exploding
    summary = mcp_server.simulate(nx=5, ny=9999, r_min_frac=0.6)
    assert 20 <= summary["grid"]["nx"] <= 250
    assert 20 <= summary["grid"]["ny"] <= 250


def test_optimize_structure_returns_best_config():
    start = mcp_server.simulate(nx=45, ny=52, r_min_frac=0.5)["flux_std"]
    r = mcp_server.optimize_structure(kind="flux_std", method="local", maxiter=6, nx=45, ny=52)
    assert r["kind"] == "flux_std"
    assert r["n_eval"] > 0
    assert r["objective"] <= start + 1e-9           # optimiser doesn't worsen the start
    assert "a_gamma" in r
    assert "v_gamma" in r


def test_fit_tracer_kinetics_recovers_flow():
    from perfusion.core import tkm

    t = np.linspace(0, 10, 200)
    ca = tkm.gamma_variate_aif(t, t0=0.5, A=3.0, B=1.0)
    conc = tkm.one_compartment(t, ca, Fp=1.2, vp=0.15)   # synthesise a known-truth curve
    r = mcp_server.fit_kinetics(t.tolist(), ca.tolist(), conc.tolist(), model="1CM")
    assert r["model"] == "1CM"
    assert r["Fp"] == pytest.approx(1.2, rel=0.1)         # recovers the ground truth
    assert r["vp"] == pytest.approx(0.15, rel=0.2)


def test_model_info_shape():
    info = mcp_server.model_info()
    assert info["name"] == "perfusion"
    assert {"pressure", "flux", "tof"} <= set(info["fields"])


def test_sweep_trend_is_monotone_in_p_in():
    r = mcp_server.sweep("p_in", [5.0, 10.0, 20.0], nx=40, ny=50, r_min_frac=0.6)
    assert r["param"] == "p_in"
    assert len(r["points"]) == 3
    inflow = [p["arterial_inflow"] for p in r["points"]]
    assert inflow[0] < inflow[1] < inflow[2]        # inflow rises with driving pressure


def test_sweep_rejects_unknown_param():
    with pytest.raises(ValueError, match="param"):
        mcp_server.sweep("bogus", [1, 2])


def test_server_registers_expected_tools():
    pytest.importorskip("mcp")
    import asyncio

    server = mcp_server.build_server()
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert {"run_simulation", "run_sweep", "perfusion_parameter_maps",
            "direct_transit_metrics", "oxygen_extraction_map",
            "heterogeneous_perfusion_lesion", "anisotropic_perfusion_bed",
            "data_constrained_tree_growth", "strahler_morphometry",
            "multicompartment_perfusion",
            "adapt_vessel_network", "uncertainty_quantification",
            "sobol_sensitivity_indices", "calibrate_parameter_to_flow",
            "describe_model"} <= tools


def test_calibrate_to_flow_recovers_value():
    import json

    # target inflow from the default model, then recover the permeability
    base = mcp_server.simulate(nx=45, ny=52, r_min_frac=0.5)
    target = base["arterial_inflow"]
    r = mcp_server.calibrate_to_flow(target, param="permeability", lo=0.2, hi=5.0,
                                     nx=45, ny=52, r_min_frac=0.5)
    json.dumps(r)
    assert r["recovered_value"] == pytest.approx(1.0, rel=0.2)   # default K = 1
    assert r["achieved"] == pytest.approx(target, rel=1e-2)


def test_sobol_sensitivity_indices():
    import json

    r = mcp_server.sobol_sensitivity(n_samples=32, nx=45, ny=52)
    json.dumps(r)
    assert r["fit_r2"] > 0.8
    assert r["first_order"]["p_in"] == max(r["first_order"].values())
    assert r["total_effect"]["p_in"] >= r["first_order"]["p_in"] - 1e-9


def test_uncertainty_ranks_inlet_pressure():
    import json

    r = mcp_server.uncertainty(n_samples=16, nx=45, ny=52)
    json.dumps(r)
    assert "arterial_inflow" in r["qoi_names"]
    assert r["std"]["arterial_inflow"] > 0
    assert r["inflow_sensitivity"]["p_in"] > 0.5     # inflow driven by inlet pressure


def test_adapt_network_moves_toward_murray():
    import json

    r = mcp_server.adapt_network(target_exponent=3.0, rate=0.05, n_iter=60,
                                 nx=45, ny=52, r_min_frac=0.5)
    json.dumps(r)
    assert r["moved_toward_target"] is True
    assert abs(r["murray_final"] - 3.0) < abs(r["murray_start"] - 3.0)


def test_multicompartment_percolates_and_conserves():
    import json

    r = mcp_server.multicompartment(n=3, beta=1.0, nx=45, ny=52, r_min_frac=0.5)
    json.dumps(r)
    assert r["n_compartments"] == 3
    # capillary perfusion equals the arterial delivery (steady conservation)
    assert r["capillary_perfusion"] == pytest.approx(r["arterial_inflow"], rel=1e-2)
    # pressure cascades arteriole (0) → venule (n−1)
    p = r["compartment_mean_pressure"]
    assert p[0] > p[-1]


def test_heterogeneous_lesion_starves_local_flux():
    import json

    r = mcp_server.heterogeneous_perfusion(lesion_k=0.02, nx=50, ny=60, r_min_frac=0.5)
    json.dumps(r)
    assert r["flux_in_lesion"] < r["flux_outside"]     # low-K lesion is under-perfused
    assert 0.0 <= r["starvation_ratio"] < 1.0


def test_strahler_morphometry_reports_ratios():
    import json

    r = mcp_server.tree_morphometry(kind="artery", seed=0)
    json.dumps(r)
    assert r["max_order"] >= 3
    assert 1.5 < r["bifurcation_ratio"] < 5.0
    assert r["diameter_ratio"] > 1.0


def test_data_constrained_growth_concentrates_terminals():
    import json

    r = mcp_server.data_constrained_growth(hotspot_weight=10.0, n_points=500, seed=0)
    json.dumps(r)
    assert r["n_terminals"] > 0
    # the tree grew denser than area alone would give inside the demand hotspot
    assert r["concentration_ratio"] > 1.2


def test_anisotropic_bed_channels_flux_along_high_k_axis():
    import json

    iso = mcp_server.anisotropic_perfusion(anisotropy=1.0, nx=45, ny=52, r_min_frac=0.5)
    aniso = mcp_server.anisotropic_perfusion(anisotropy=8.0, nx=45, ny=52, r_min_frac=0.5)
    json.dumps(aniso)
    # raising Ky pushes relatively more flux through the y-faces than the isotropic bed
    iso_ratio = iso["mean_abs_flux_y"] / iso["mean_abs_flux_x"]
    aniso_ratio = aniso["mean_abs_flux_y"] / aniso["mean_abs_flux_x"]
    assert aniso_ratio > iso_ratio


def test_oxygen_extraction_increases_with_rate():
    import json

    lo = mcp_server.oxygen_extraction(k=0.3, nx=45, ny=52, r_min_frac=0.5)
    hi = mcp_server.oxygen_extraction(k=3.0, nx=45, ny=52, r_min_frac=0.5)
    json.dumps(lo)
    assert 0.0 < lo["oef"] < hi["oef"] < 1.0
    assert 0.0 <= hi["min_saturation"] <= hi["mean_saturation"] <= 1.0 + 1e-9


def test_direct_transit_metrics_sweep_order():
    import json

    r = mcp_server.direct_transit(rho_c=0.0, p_in=10.0, r_min_frac=0.5)
    json.dumps(r)
    assert r["n_connections"] > 0
    assert 0.0 < r["reachable_fraction"] <= 1.0
    # bolus reaches arteries before connections before veins
    assert r["mean_arrival_arterial"] < r["mean_arrival_connection"]
    assert r["mean_arrival_connection"] < r["mean_arrival_venous"]


def test_perfusion_maps_summarises_active_tissue():
    import json

    r = mcp_server.perfusion_maps(nx=45, ny=52, n_frames=16, r_min_frac=0.5)  # shallow -> fast
    json.dumps(r)
    assert r["n_frames"] == 16
    assert r["Fp"]["max"] > 0                       # some perfused tissue
    assert r["vp"]["mean"] is not None

    d = mcp_server.perfusion_maps(nx=45, ny=52, n_frames=16, r_min_frac=0.5, method="deconv")
    json.dumps(d)
    assert d["method"] == "deconv"
    assert {"Fp", "CBV", "MTT"} <= set(d)           # deconvolution reports transit time
    assert d["MTT"]["mean"] is not None


def test_simulate_is_json_serialisable():
    import json

    json.dumps(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6))
    assert np.isfinite(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6)["pressure_max"])


def test_prune_tree_reduces_generations():
    r = mcp_server.prune_tree(kind="artery", steps=1, r_min_frac=0.3, seed=0)
    assert r["removed_segments"] > 0
    assert r["after"]["segments"] < r["before"]["segments"]
    assert r["after"]["max_order"] <= r["before"]["max_order"]


@pytest.mark.parametrize(("dim", "shape"), [(2, "kidney"), (3, "hemisphere")])
def test_generate_trees_fills_tissue_and_conserves_mass(dim, shape):
    r = mcp_server.generate_trees(dim=dim, shape=shape, generations=3, fill="cco",
                                  n_terminals=15)
    assert r["artery"]["segments"] > 7
    assert r["inflow"] > 0
    assert r["inflow"] == pytest.approx(r["outflow"], rel=1e-6)
    h = r["homogeneity"]                       # flux + time-of-flight, in 2-D and 3-D
    assert np.isfinite(h["tof_std"])
    assert h["flux_mean"] > 0


def test_generate_trees_uses_anatomical_roots_and_solves():
    r = mcp_server.generate_trees(dim=2, shape="kidney", generations=3, fill="cco",
                                  n_terminals=15)
    assert r["inflow"] > 0
    assert r["inflow"] == pytest.approx(r["outflow"], rel=1e-6)
