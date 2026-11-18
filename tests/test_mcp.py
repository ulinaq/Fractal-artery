"""Tests for the MCP server that exposes the perfusion engine.

The tool *logic* (``simulate`` / ``model_info``) needs no ``mcp`` package; the
server construction does (``pytest.importorskip``).
"""

from pathlib import Path

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
            "tissue_perfusion_analysis", "optimize_trees_in_tissue",
            "tissue_uncertainty_calibration", "direct_model_in_tissue",
            "export_tissue_fields", "lesion_in_tissue", "compare_scenarios",
            "experiment_in_tissue", "fit_dce_curve",
            "anastomosis_rescue", "shear_thinning_blood", "describe_model"} <= tools


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


@pytest.mark.parametrize("dim", [2, 3])
def test_tissue_analysis_runs_the_same_chain_in_2d_and_3d(dim):
    r = mcp_server.tissue_analysis(dim=dim, shape="kidney", generations=3, n_terminals=15,
                                   n_frames=12, adapt=dim == 3)
    assert r["dim"] == dim
    assert r["inflow"] > 0
    assert r["transit"]["mtt"] > 0
    assert 0.2 < r["oxygen"]["oef"] < 0.8           # organ-volume-scaled Damköhler rate
    assert r["tracer"]["Fp"] > 0
    assert r["tracer"]["peak_concentration"] > 0
    if dim == 3:
        a = r["adaptation"]
        assert abs(a["murray_final"] - 3) <= abs(a["murray_start"] - 3)


@pytest.mark.parametrize("dim", [2, 3])
def test_tissue_analysis_multicompartment_bed(dim):
    r = mcp_server.tissue_analysis(dim=dim, shape="kidney", generations=3, n_terminals=15,
                                   n_frames=8, compartments=3, beta=1.0)
    mc = r["multicompartment"]
    assert mc["n"] == 3
    assert mc["capillary_perfusion"] > 0
    p = mc["compartment_mean_pressure"]
    assert p[0] > p[1] > p[2]                        # arteriole → venule cascade


@pytest.mark.parametrize("dim", [2, 3])
def test_tissue_sweep_uq_and_calibration_in_the_organ(dim):
    r = mcp_server.tissue_sweep(dim=dim, shape="kidney", generations=3, n_terminals=15,
                                n_samples=6, permeability_range=(0.5, 2.0))
    assert r["dim"] == dim
    sens = r["uq"]["inflow_sensitivity"]
    assert sens["p_in"] > 0
    assert sens["mu"] < 0
    s = mcp_server.tissue_sweep(dim=dim, shape="kidney", generations=3, n_terminals=15,
                                n_samples=8, mu_range=None, method="sobol",
                                calibrate_target=None)
    assert s["uq"]["fit_r2"] > 0.9
    base = mcp_server.generate_trees(dim=dim, shape="kidney", generations=3, fill="cco",
                                     n_terminals=15)["inflow"]
    c = mcp_server.tissue_sweep(dim=dim, shape="kidney", generations=3, n_terminals=15,
                                p_in_range=None, mu_range=None, calibrate_param="p_in",
                                calibrate_target=2 * base, calibrate_bounds=(1, 40))
    assert "uq" not in c
    assert c["calibration"]["recovered_value"] == pytest.approx(20.0, rel=1e-3)


@pytest.mark.parametrize("dim", [2, 3])
def test_optimize_tissue_trees_improves_in_the_organ(dim):
    r = mcp_server.optimize_tissue_trees(dim=dim, shape="kidney", generations=3,
                                         n_terminals=15, maxiter=4)
    assert r["dim"] == dim
    assert r["n_eval"] > 0
    assert r["best"] <= r["start"]
    assert ("twist" in r["free"]) == (dim == 3)
    assert 1.0 <= r["values"]["a_gamma"] <= 1.9


@pytest.mark.parametrize("dim", [2, 3])
def test_direct_model_in_tissue(dim):
    """The direct-connection model on organ-grown trees (2-D and 3-D): mass balance,
    every arterial terminal linked, and the bolus reaches the venous outlet."""
    out = mcp_server.tissue_direct(dim=dim, shape="kidney", generations=3,
                                   n_terminals=15, rho_c=0.0)
    assert out["dim"] == dim
    assert out["model"] == "direct"
    assert out["arterial_inflow"] > 0
    assert out["venous_outflow"] == pytest.approx(out["arterial_inflow"], rel=1e-6)
    assert out["n_connections"] == out["n_terminals"]["artery"]
    assert out["connection_length"]["mean"] > 0
    assert out["transit"]["mtt_to_outlet"] > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_export_tissue_fields(tmp_path, dim):
    out = mcp_server.tissue_export(str(tmp_path), dim=dim, shape="kidney", generations=3,
                                   n_terminals=10)
    assert out["inflow"] > 0
    for path in out["files"].values():
        assert Path(path).stat().st_size > 0
    header = Path(out["files"]["fields.csv"]).read_text().splitlines()[0]
    assert header == ("x,y,z," if dim == 3 else "x,y,") + "pressure,flux,tof"


def test_export_tissue_fields_with_a_3d_video(tmp_path):
    pytest.importorskip("imageio")
    out = mcp_server.tissue_export(str(tmp_path), dim=3, shape="kidney", generations=3,
                                   n_terminals=10, video=True, n_frames=8)
    gif = Path(out["files"]["tracer.gif"])
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


@pytest.mark.parametrize("dim", [2, 3])
def test_lesion_in_tissue_cuts_local_perfusion(dim):
    """A low-K ball in an organ (disc in 2-D, sphere in 3-D) throttles the flow
    through it; the milder lesion leaves more."""
    strong = mcp_server.tissue_lesion(dim=dim, shape="kidney", generations=3,
                                      n_terminals=15, permeability=0.02)
    mild = mcp_server.tissue_lesion(dim=dim, shape="kidney", generations=3,
                                    n_terminals=15, permeability=0.5)
    assert strong["lesion_cells"] > 0
    assert strong["remaining_fraction"] < 0.85
    assert 1.0 > mild["remaining_fraction"] > strong["remaining_fraction"]
    assert strong["inflow_lesion"] <= strong["inflow_healthy"] * (1 + 1e-9)
    with pytest.raises(ValueError, match="center needs"):
        mcp_server.tissue_lesion(dim=dim, center=[1.0] * (5 - dim))


@pytest.mark.parametrize("dim", [2, 3])
def test_compare_scenarios_in_an_organ(dim):
    """A/B on the same trees: raising p_in scales the (linear) flow; no change is a
    zero delta; a lesion lowers the local flux; bad keys are rejected."""
    up = mcp_server.tissue_compare(dim=dim, shape="kidney", generations=3, n_terminals=12,
                                   baseline={"p_in": 10.0}, change={"p_in": 12.0})
    rows = {r["key"]: r for r in up["summary"]}
    assert rows["arterial_inflow"]["rel"] == pytest.approx(0.2, rel=1e-6)   # linear model
    assert up["same_grid"] is True
    same = mcp_server.tissue_compare(dim=dim, shape="kidney", generations=3, n_terminals=12)
    assert same["fields"]["flux"]["mean_abs_diff"] == 0.0
    les = mcp_server.tissue_compare(dim=dim, shape="kidney", generations=3, n_terminals=12,
                                    change={"lesion": {"permeability": 0.02}})
    assert {r["key"]: r for r in les["summary"]}["arterial_inflow"]["diff"] <= 1e-12
    with pytest.raises(ValueError, match="unknown scenario keys"):
        mcp_server.tissue_compare(dim=dim, change={"pressure": 3})


@pytest.mark.parametrize("dim", [2, 3])
def test_experiment_in_tissue_sweep(dim):
    out = mcp_server.tissue_experiment(param="p_in", values=[5, 10, 20], dim=dim,
                                       shape="kidney", generations=3, n_terminals=12)
    q = [r["arterial_inflow"] for r in out["rows"]]
    assert q[0] == pytest.approx(0.5 * q[1], rel=1e-9)        # linear in the drop
    assert q[2] == pytest.approx(2.0 * q[1], rel=1e-9)
    k = mcp_server.tissue_experiment(param="permeability", values=[0.5, 2], dim=dim,
                                     shape="kidney", generations=3, n_terminals=12)
    assert k["rows"][0]["arterial_inflow"] < k["rows"][1]["arterial_inflow"]
    with pytest.raises(ValueError, match="param must be"):
        mcp_server.tissue_experiment(param="mu")


@pytest.mark.parametrize("dim", [2, 3])
def test_experiment_in_tissue_grid(dim):
    out = mcp_server.tissue_experiment(param="p_in", values=[5, 10], dim=dim,
                                       shape="kidney", generations=3, n_terminals=12,
                                       param2="permeability", values2=[0.5, 1, 2])
    assert out["param2"] == "permeability"
    assert [(r["value"], r["value2"]) for r in out["rows"]] == [
        (a, b) for a in (5.0, 10.0) for b in (0.5, 1.0, 2.0)]
    s = out["sensitivity"]["arterial_inflow"]
    assert s["p_in"] == pytest.approx(1.0, abs=1e-9)
    assert 0 < s["permeability"] < 1                      # the bed shares the resistance
    one = mcp_server.tissue_experiment(param="p_in", values=[5, 10], dim=dim,
                                       shape="kidney", generations=3, n_terminals=12)
    assert one["sensitivity"]["arterial_inflow"]["p_in"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="differ"):
        mcp_server.tissue_experiment(param="p_in", param2="p_in", values2=[1, 2])
    with pytest.raises(ValueError, match="at most 48"):
        mcp_server.tissue_experiment(param="p_in", values=list(range(1, 9)),
                                     param2="permeability", values2=list(range(1, 8)))


@pytest.mark.parametrize("dim", [2, 3])
def test_fit_dce_curve_recovers_a_synthetic_truth(dim):
    out = mcp_server.tissue_dce_fit(dim=dim, shape="kidney", generations=3, n_terminals=12,
                                    truth={"p_in": 15.0, "porosity": 0.25}, n_samples=30)
    assert out["synthetic"]["truth"]["p_in"] == 15.0
    assert out["params"]["p_in"] == pytest.approx(15.0, rel=1e-3)
    assert out["params"]["porosity"] == pytest.approx(0.25, rel=1e-3)
    assert out["r2"] > 0.999
    with pytest.raises(ValueError, match="together"):
        mcp_server.tissue_dce_fit(times=[0, 1, 2, 3], dim=dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_anastomosis_rescue_in_an_organ(dim):
    out = mcp_server.tissue_anastomosis(dim=dim, shape="kidney", generations=5, n_links=6)
    assert out["n_loops"] == out["n_links"] > 0
    assert out["candidates"]
    for row in out["candidates"]:
        assert row["tree_occluded"] < 1e-6                   # a tree starves the territory
        assert row["looped_occluded"] >= row["tree_occluded"] - 1e-9  # round-off at ~0
    assert out["best"]["looped_occluded"] > 0.01               # loops bring some back
    with pytest.raises(ValueError, match="segment index"):
        mcp_server.tissue_anastomosis(dim=dim, occlude=10**6)


@pytest.mark.parametrize("dim", [2, 3])
def test_shear_thinning_blood_in_an_organ(dim):
    out = mcp_server.tissue_rheology(dim=dim, shape="kidney", generations=4,
                                     pressures=(5.0, 10.0, 20.0))
    assert out["dim"] == dim
    assert out["flow_exponent_newtonian"] == pytest.approx(1.0, abs=1e-9)
    assert out["flow_exponent_carreau"] > 1.05             # thinning: Q grows faster than Δp
    ratios = [r["flow_ratio"] for r in out["rows"]]
    assert all(0 < q < 1 for q in ratios)                  # thicker than the high-shear mu
    assert ratios == sorted(ratios)                        # ... less so at higher pressure
    assert all(r["viscosity_factor_min"] >= 1.0 for r in out["rows"])
    with pytest.raises(ValueError, match="positive"):
        mcp_server.tissue_rheology(dim=dim, pressures=(0.0, 5.0))
