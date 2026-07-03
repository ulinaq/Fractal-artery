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


def test_model_info_shape():
    info = mcp_server.model_info()
    assert info["name"] == "perfusion"
    assert {"pressure", "flux", "tof"} <= set(info["fields"])


def test_server_registers_expected_tools():
    pytest.importorskip("mcp")
    import asyncio

    server = mcp_server.build_server()
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert {"run_simulation", "describe_model"} <= tools


def test_simulate_is_json_serialisable():
    import json

    json.dumps(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6))
    assert np.isfinite(mcp_server.simulate(nx=40, ny=40, r_min_frac=0.6)["pressure_max"])
