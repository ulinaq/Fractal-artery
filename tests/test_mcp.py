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
    assert {"run_simulation", "run_sweep", "describe_model"} <= tools


def test_simulate_is_json_serialisable():
    import json

    json.dumps(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6))
    assert np.isfinite(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6)["pressure_max"])
