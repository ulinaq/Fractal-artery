"""MCP server exposing the perfusion engine as tools (optional ``mcp`` extra).

Any Model Context Protocol client -- Claude Desktop, Claude Code, an agent -- can
then run blood-flow/perfusion simulations and read the model description as
first-class tools. Start it with ``uv run --extra mcp perfusion-mcp`` (stdio
transport); the repo's ``.mcp.json`` wires it up for Claude Code automatically.

The tool *logic* lives in plain functions (:func:`simulate`, :func:`model_info`)
so it is unit-testable without the ``mcp`` package; the MCP tools are thin
wrappers. Depends only on the pure ``core`` engine + ``io`` summary -- no API,
store, or web state.
"""

from __future__ import annotations

import numpy as np

from .core.darcy import Grid
from .core.parameters import Parameters
from .core.pipeline import run_hybrid
from .io import results_summary

# The default trees are generated to fit this coordinate window (see the bundled
# reference in tests/data); keep lx/ly = the window's width/height so terminals
# land on the grid. Only the resolution (nx, ny) is caller-tunable.
_DOMAIN = (-70.0, 70.0, 0.0, 160.0)
_LX, _LY = _DOMAIN[1] - _DOMAIN[0], _DOMAIN[3] - _DOMAIN[2]


def simulate(p_in: float = 10.0, p_out: float = 0.0, nx: int = 120, ny: int = 140,
             r_min_frac: float = 0.15, nonlinear: bool = False) -> dict:
    """Run a hybrid perfusion simulation on fractal trees; return a JSON summary.

    Couples fractal arterial/venous trees to a Darcy capillary bed and solves the
    flow, returning scalar results (arterial inflow, venous outflow, and
    pressure / flux / time-of-flight statistics). ``nx``/``ny`` set the Darcy grid
    resolution; ``r_min_frac`` controls how deep the trees branch (larger ->
    shallower, faster).
    """
    nx, ny = int(np.clip(nx, 20, 250)), int(np.clip(ny, 20, 250))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN, nonlinear=nonlinear)
    summary = results_summary(result)
    summary["grid"] = {"nx": nx, "ny": ny}
    return summary


_SWEEPABLE = ("p_in", "p_out", "r_min_frac")


def sweep(param: str, values, **base) -> dict:
    """Run :func:`simulate` across ``values`` of one parameter; return the trend.

    ``param`` is one of ``p_in``, ``p_out``, ``r_min_frac``. Each point reports the
    arterial inflow and mean flux / time-of-flight, so you can see how perfusion
    responds to the parameter.
    """
    if param not in _SWEEPABLE:
        raise ValueError(f"param must be one of {_SWEEPABLE}")
    points = []
    for v in values:
        s = simulate(**{**base, param: float(v)})
        points.append({"value": float(v),
                       "arterial_inflow": s["arterial_inflow"],
                       "flux_mean": s["flux_mean"], "tof_mean": s["tof_mean"]})
    return {"param": param, "points": points}


def model_info() -> dict:
    """Describe what the perfusion model computes and the fields it returns."""
    return {
        "name": "perfusion",
        "summary": ("Multiscale blood circulation & perfusion: a 1-D vascular "
                    "network (Poiseuille flow) coupled to a Darcy porous-medium "
                    "capillary bed, with tracer transport and tracer-kinetic models."),
        "boundary_conditions": ("Dirichlet: p_in at the arterial inlet node(s), "
                                "p_out at the venous outlet node(s); any number of each."),
        "fields": {
            "pressure": "capillary-bed pressure field (nx, ny)",
            "flux": "Darcy flux magnitude (nx, ny)",
            "tof": "time-of-flight of blood through each cell (nx, ny)",
        },
        "papers": {
            "A": "nonlinear hybrid model (network + dual Darcy)",
            "B": "tracer flow in the hybrid model",
            "C": "direct-connection microcirculation + tracer-kinetic models",
            "D": "vascular-structure optimisation (flux vs. time-of-flight)",
        },
    }


def build_server():
    """Construct the FastMCP server (imports ``mcp`` lazily)."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("perfusion")

    @server.tool()
    def run_simulation(p_in: float = 10.0, p_out: float = 0.0, nx: int = 120,
                       ny: int = 140, r_min_frac: float = 0.15,
                       nonlinear: bool = False) -> dict:
        """Run a hybrid blood-flow + Darcy perfusion simulation on fractal
        arterial/venous trees and return a scalar summary (arterial inflow,
        venous outflow, pressure/flux/time-of-flight statistics)."""
        return simulate(p_in=p_in, p_out=p_out, nx=nx, ny=ny,
                        r_min_frac=r_min_frac, nonlinear=nonlinear)

    @server.tool()
    def run_sweep(param: str, values: list[float], p_in: float = 10.0,
                  p_out: float = 0.0, nx: int = 90, ny: int = 110,
                  r_min_frac: float = 0.15) -> dict:
        """Sweep one parameter ('p_in', 'p_out' or 'r_min_frac') over `values` and
        return the arterial inflow / mean flux / mean time-of-flight at each — a
        quick way to see how perfusion responds to a parameter."""
        base = {"p_in": p_in, "p_out": p_out, "nx": nx, "ny": ny, "r_min_frac": r_min_frac}
        return sweep(param, values, **base)

    @server.tool()
    def describe_model() -> dict:
        """Describe the perfusion model: what it computes, its boundary
        conditions, and the fields it returns."""
        return model_info()

    return server


def main() -> None:
    """Run the MCP server over stdio (``perfusion-mcp``)."""
    build_server().run()


if __name__ == "__main__":
    main()
