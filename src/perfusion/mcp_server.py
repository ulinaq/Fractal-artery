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
             r_min_frac: float = 0.15, nonlinear: bool = False,
             max_iter: int = 100, model: str = "hybrid", rho_c: float = 0.0) -> dict:
    """Run a perfusion simulation on fractal trees; return a JSON summary.

    ``model="hybrid"`` (default) couples arterial/venous trees to a Darcy capillary
    bed; ``model="direct"`` closes the trees with resistive terminal connections
    (Paper C) instead — ``rho_c`` is the connection search radius (0 = nearest vein).
    ``nx``/``ny`` set the Darcy grid resolution; ``r_min_frac`` controls branch depth
    (larger -> shallower); ``max_iter`` caps the nonlinear fixed-point iteration.
    """
    nx, ny = int(np.clip(nx, 20, 250)), int(np.clip(ny, 20, 250))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    if model == "direct":
        from .core.direct import connect_terminals, solve_direct
        from .core.geometry import arterial_tree, venous_tree

        art, ven = arterial_tree(prm), venous_tree(prm)
        conns = connect_terminals(art, ven, rho_c=float(rho_c), viscosity=prm.mu)
        sol = solve_direct(art, ven, conns, prm.p_in, prm.p_out, viscosity=prm.mu)
        perfusion = float(np.abs(sol.q_c).sum())
        return {"model": "direct", "arterial_inflow": perfusion, "venous_outflow": perfusion,
                "n_connections": len(conns)}
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN, nonlinear=nonlinear,
                        max_iter=int(max_iter))
    summary = results_summary(result)
    summary["grid"] = {"nx": nx, "ny": ny}
    summary["model"] = "hybrid"
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


def optimize_structure(kind: str = "tof_std", method: str = "local", maxiter: int = 30,
                       nx: int = 70, ny: int = 80) -> dict:
    """Optimise the fractal-tree configuration to homogenise perfusion (Paper D).

    Minimises flux- or time-of-flight heterogeneity over the tree's branching
    asymmetries; returns the best objective and the key resulting parameters.
    """
    from .core.darcy import Grid
    from .optimize import optimize_tree

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    r = optimize_tree(kind=kind, grid=grid, domain_limit=_DOMAIN, method=method,
                      maxiter=int(maxiter))
    return {"kind": r.kind, "objective": r.objective, "n_eval": r.n_eval,
            "success": r.success, "free": r.free,
            "a_gamma": r.alpha[0], "v_gamma": r.alpha[1], "exponent": r.alpha[8]}


def fit_kinetics(t, aif, concentration, model: str = "2CXM") -> dict:
    """Fit a tracer-kinetic model to a tissue concentration curve (Paper C)."""
    from .core import tkm

    r = tkm.fit(np.asarray(t, float), np.asarray(aif, float),
                np.asarray(concentration, float), model=model)

    def num(x):
        return float(x) if x is not None and np.isfinite(x) else None

    return {"model": model.upper(), "Fp": num(r.Fp), "vp": num(r.vp),
            "PS": num(r.PS), "ve": num(r.ve)}


def pulsatile_response(r_total: float, compliance_total: float, p_mean: float = 10.0,
                       p_out: float = 0.0, pulse: float = 0.5, period: float = 1.0) -> dict:
    """Windkessel pulsatile response: how much a compliant vasculature damps the
    inlet pulse before it reaches the tissue."""
    from .core import transient

    dt = period / 100.0
    _, p_in = transient.pulsatile_inlet(p_mean, pulse * (p_mean - p_out), period, dt, n_cycles=4)
    wk = transient.windkessel(p_in, r_total, compliance_total, dt, p_out=p_out)
    last = slice(-100, None)

    def puls(a):
        a = np.asarray(a[last], float)
        return float((a.max() - a.min()) / max(a.mean(), 1e-12))

    return {"R": r_total, "C": compliance_total,
            "inlet_pulsatility": puls(p_in), "tissue_pulsatility": puls(wk["q_tissue"])}


def perfusion_maps(nx: int = 70, ny: int = 80, n_frames: int = 24,
                   p_in: float = 10.0, p_out: float = 0.0, r_min_frac: float = 0.15,
                   method: str = "semi", oi: float = 0.1) -> dict:
    """Synthesise a contrast bolus on the fractal trees and reduce it to voxel-wise
    perfusion maps (Paper B); return summary statistics over the active tissue (the
    app renders the full 2-D fields).

    ``method='semi'`` gives max-slope ``Fp`` + AUC ``vp``; ``method='deconv'`` gives
    residue-deconvolution ``Fp``/``CBV``/``MTT`` (``oi`` = SVD truncation)."""
    from .core import tkm, transport

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN)

    finite = result.tof[np.isfinite(result.tof)]
    t_char = float(np.percentile(finite, 80)) if finite.size else 5.0
    duration = 5.0 * max(t_char, 1e-3)
    n_steps = max(n_frames, 500)
    dt = duration / n_steps
    t = np.arange(n_steps) * dt
    aif = tkm.gamma_variate_aif(t, t0=0.08 * duration, A=3.0, B=max(0.06 * duration, dt))
    peak = float(aif.max())
    if peak > 0:
        aif = aif / peak
    full = transport.transport_forward(grid, result.fx, result.fy, result.sources,
                                       result.solution.terminal_flux, aif, dt, theta=1.0)
    stride = max(1, n_steps // n_frames)
    frames, times = full[::stride][:n_frames], t[::stride][:n_frames]
    ca = aif[::stride][:n_frames]
    if method == "deconv":
        m = tkm.deconvolution_maps(frames, times, ca, mask=grid.mask, oi=float(oi))
    else:
        m = tkm.kinetic_maps(frames, times, ca, mask=grid.mask)

    def stat(a):
        a = a[np.isfinite(a)]
        if not a.size:
            return {"mean": None, "max": None}
        return {"mean": float(a.mean()), "max": float(a.max())}

    out = {"grid": {"nx": nx, "ny": ny}, "n_frames": len(times), "method": method}
    out.update({k: stat(v) for k, v in m.items()})
    return out


def direct_transit(rho_c: float = 0.0, p_in: float = 10.0, p_out: float = 0.0,
                   r_min_frac: float = 0.15) -> dict:
    """Network-transit kinetics for the direct-connection model (Paper C).

    Solves the direct model on the fractal trees and times a bolus along the
    vasculature (``τ = volume/|flow|`` accumulated arterial → connection → venous).
    Returns transit-time statistics: mean arrival by element class and the overall
    mean transit time (MTT) to the venous outlet."""
    from .core import network_transit as nt
    from .core.direct import connect_terminals, solve_direct
    from .core.geometry import arterial_tree, venous_tree

    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    art, ven = arterial_tree(prm), venous_tree(prm)
    conns = connect_terminals(art, ven, rho_c=float(rho_c), viscosity=prm.mu)
    sol = solve_direct(art, ven, conns, prm.p_in, prm.p_out, viscosity=prm.mu)
    anim = nt.transit_animation(art, ven, conns, sol, n_frames=2)
    arr, kind = anim["arrivals"], anim["kind"]
    t_ven = nt.arrival_times(art, ven, conns, sol)[1]

    def mean_arr(sel):
        a = arr[sel & np.isfinite(arr)]
        return float(a.mean()) if a.size else None

    outlets = np.array([int(n) for n in ven.inlet_nodes])
    mtt = t_ven[outlets]
    mtt = mtt[np.isfinite(mtt)]
    return {"n_connections": len(conns),
            "reachable_fraction": float(np.mean(np.isfinite(arr))),
            "mean_arrival_arterial": mean_arr(kind == 0),
            "mean_arrival_connection": mean_arr(kind == 2),
            "mean_arrival_venous": mean_arr(kind == 1),
            "mtt_to_outlet": float(mtt.mean()) if mtt.size else None}


def oxygen_extraction(k: float = 1.0, diffusivity: float = 0.0, c_art: float = 1.0,
                      nx: int = 90, ny: int = 110, p_in: float = 10.0, p_out: float = 0.0,
                      r_min_frac: float = 0.15) -> dict:
    """Steady oxygen / solute extraction on the Darcy field (Krogh/Kety).

    Runs the hybrid model, delivers a blood-borne solute at the feeds and consumes
    it in tissue at dimensionless Damköhler rate ``k``. Returns the global oxygen
    extraction fraction (OEF) and mean tissue saturation. ``k`` sets extraction vs.
    perfusion: OEF ≈ 1−e^{−k} in the well-mixed limit, lower with transit heterogeneity."""
    from .core import oxygen

    nx, ny = int(np.clip(nx, 30, 200)), int(np.clip(ny, 30, 200))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN)
    ox = oxygen.oxygen_from_result(result, c_art=c_art, k=k, diffusivity=diffusivity)
    sat = ox.saturation
    return {"k": k, "oef": ox.oef, "delivery": ox.delivery, "consumption": ox.consumption,
            "mean_saturation": float(np.nanmean(sat)), "min_saturation": float(np.nanmin(sat))}


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
    def optimize_tree_config(kind: str = "tof_std", method: str = "local",
                             maxiter: int = 30, nx: int = 70, ny: int = 80) -> dict:
        """Optimise the fractal arterial/venous tree to homogenise perfusion
        (Paper D): minimise flux- or time-of-flight heterogeneity. `kind` in
        {flux_std, flux_range, flux_l2, flux_mean, tof_std, tof_max}; `method` is
        'local' (Nelder-Mead) or 'global' (differential evolution). Returns the
        best objective and key tree parameters. Each step is a full solve — keep
        the grid coarse and `maxiter` modest."""
        return optimize_structure(kind=kind, method=method, maxiter=maxiter, nx=nx, ny=ny)

    @server.tool()
    def fit_tracer_kinetics(t: list[float], aif: list[float], concentration: list[float],
                            model: str = "2CXM") -> dict:
        """Fit a tracer-kinetic model (MS / 1CM / 2CUM / 2CXM) to a tissue
        concentration curve, given uniform time samples `t`, the arterial input
        function `aif`, and the tissue curve `concentration`. Returns plasma flow
        Fp, plasma volume vp, and PS / ve for the two-compartment models."""
        return fit_kinetics(t, aif, concentration, model=model)

    @server.tool()
    def pulsatile_windkessel(r_total: float, compliance_total: float, p_mean: float = 10.0,
                             p_out: float = 0.0, pulse: float = 0.5, period: float = 1.0) -> dict:
        """Lumped Windkessel transient response: given a total resistance `r_total`
        and compliance `compliance_total`, drive a pulsatile inlet and report the
        inlet vs tissue-flow pulsatility — i.e. how much vessel compliance damps the
        pulse (the gateway to transient/pulsatile perfusion)."""
        return pulsatile_response(r_total, compliance_total, p_mean, p_out, pulse, period)

    @server.tool()
    def perfusion_parameter_maps(nx: int = 70, ny: int = 80, n_frames: int = 24,
                                 p_in: float = 10.0, p_out: float = 0.0,
                                 r_min_frac: float = 0.15, method: str = "semi",
                                 oi: float = 0.1) -> dict:
        """Advect a synthetic contrast bolus through the fractal trees and reduce it
        to voxel-wise perfusion maps. method='semi' -> max-slope Fp + AUC blood
        volume vp; method='deconv' -> residue-deconvolution Fp / CBV / MTT (oi = SVD
        truncation fraction). Returns mean/max of each map over the active tissue.
        Each call is a full hybrid solve + transport — keep the grid coarse."""
        return perfusion_maps(nx=nx, ny=ny, n_frames=n_frames, p_in=p_in, p_out=p_out,
                              r_min_frac=r_min_frac, method=method, oi=oi)

    @server.tool()
    def direct_transit_metrics(rho_c: float = 0.0, p_in: float = 10.0, p_out: float = 0.0,
                               r_min_frac: float = 0.15) -> dict:
        """Network-transit kinetics for the direct-connection model: solve it on the
        fractal trees and time a bolus along the vasculature (transit time =
        volume/flow, accumulated arterial -> capillary connection -> venous).
        Returns the mean bolus arrival time by element class and the overall mean
        transit time to the venous outlet. `rho_c` is the connection search radius."""
        return direct_transit(rho_c=rho_c, p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)

    @server.tool()
    def oxygen_extraction_map(k: float = 1.0, diffusivity: float = 0.0, c_art: float = 1.0,
                              nx: int = 90, ny: int = 110, p_in: float = 10.0,
                              p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
        """Steady oxygen / solute extraction (Krogh/Kety) on the hybrid Darcy field:
        deliver a blood-borne solute at the feeds, consume it in tissue at
        dimensionless Damköhler rate `k`, and report the global oxygen-extraction
        fraction (OEF) + mean/min tissue saturation. `diffusivity` adds tissue
        diffusion. Each call is a full hybrid solve — keep the grid coarse."""
        return oxygen_extraction(k=k, diffusivity=diffusivity, c_art=c_art, nx=nx, ny=ny,
                                 p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)

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
