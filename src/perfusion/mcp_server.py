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
             max_iter: int = 100, model: str = "hybrid", rho_c: float = 0.0,
             _reuse=None) -> dict:
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
    art, ven, resp = _reuse if _reuse is not None else (None, None, None)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN, nonlinear=nonlinear,
                        max_iter=int(max_iter), art=art, ven=ven, darcy_response=resp)
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
    reuse = None
    if param != "r_min_frac" and base.get("model", "hybrid") == "hybrid":
        # p_in / p_out leave grid and terminals unchanged: factorise the Darcy bed once
        from .core.pipeline import darcy_response_for

        nx = int(np.clip(base.get("nx", 120), 20, 250))
        ny = int(np.clip(base.get("ny", 140), 20, 250))
        prm0 = Parameters(r_min_frac=base.get("r_min_frac", 0.15))
        reuse = darcy_response_for(prm0, Grid(nx=nx, ny=ny, lx=_LX, ly=_LY), _DOMAIN)
    points = []
    for v in values:
        s = simulate(**{**base, param: float(v)}, _reuse=reuse)
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


def calibrate_to_flow(target_inflow: float, param: str = "permeability",
                      lo: float = 0.1, hi: float = 10.0, nx: int = 70, ny: int = 80,
                      p_in: float = 10.0, p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
    """Inverse calibration: recover a model input (``permeability`` / ``p_in`` / ``mu``)
    so the fractal-tree model's arterial inflow matches ``target_inflow``. Returns the
    recovered value, the achieved inflow and the number of forward solves."""
    from . import inverse

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    res = inverse.calibrate(float(target_inflow), param, grid, (lo, hi),
                            qoi="arterial_inflow",
                            base={"p_in": p_in, "p_out": p_out, "r_min_frac": r_min_frac},
                            domain_limit=_DOMAIN)
    return {"param": res.param, "recovered_value": res.value, "target": res.target,
            "achieved": res.achieved, "n_eval": res.n_eval}


def uncertainty(n_samples: int = 24, p_in_range=(5.0, 20.0), mu_range=(0.8, 1.5),
                rmin_range=(0.35, 0.6), nx: int = 60, ny: int = 70,
                seed: int = 0) -> dict:
    """Monte-Carlo uncertainty quantification of the hybrid model.

    Latin-hypercube-samples the inlet pressure, viscosity and branch-depth over the
    given ranges, re-solves ``n_samples`` times, and returns each quantity of interest's
    mean/std and the first-order sensitivity (Pearson r) of arterial inflow to each
    input — a cheap ranking of what drives perfusion. Keep the grid coarse."""
    from . import uq

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    res = uq.monte_carlo(
        {"p_in": tuple(p_in_range), "mu": tuple(mu_range), "r_min_frac": tuple(rmin_range)},
        grid, n_samples=int(n_samples), base={"p_out": 0.0}, seed=int(seed),
        domain_limit=_DOMAIN)
    return {"n_samples": int(n_samples), "qoi_names": res.qoi_names,
            "mean": res.mean, "std": res.std,
            "inflow_sensitivity": res.sensitivity["arterial_inflow"]}


def sobol_sensitivity(n_samples: int = 48, p_in_range=(5.0, 20.0), mu_range=(0.8, 1.5),
                      rmin_range=(0.35, 0.6), nx: int = 60, ny: int = 70,
                      seed: int = 0) -> dict:
    """Variance-based (Sobol) sensitivity of the hybrid model via a polynomial-chaos
    surrogate. Fits a degree-2 expansion to ``n_samples`` Latin-hypercube solves and
    returns, for arterial inflow, the first-order (``S1``) and total-effect (``ST``)
    Sobol indices of each input — the rigorous variance decomposition — plus the
    surrogate fit R²."""
    from . import uq

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    res = uq.sobol_indices(
        {"p_in": tuple(p_in_range), "mu": tuple(mu_range), "r_min_frac": tuple(rmin_range)},
        grid, n_samples=int(n_samples), base={"p_out": 0.0}, seed=int(seed),
        domain_limit=_DOMAIN)
    return {"n_samples": res.n_samples, "fit_r2": res.fit_r2["arterial_inflow"],
            "first_order": res.first_order["arterial_inflow"],
            "total_effect": res.total_effect["arterial_inflow"]}


def adapt_network(target_exponent: float = 3.0, rate: float = 0.05, n_iter: int = 60,
                  nx: int = 70, ny: int = 80, p_in: float = 10.0, p_out: float = 0.0,
                  r_min_frac: float = 0.15) -> dict:
    """Structural adaptation of vessel radii toward Murray's law (reduced Pries–Secomb).

    Wall-shear-driven remodelling evolves segment radii until the ``|Q| ∝ r^k`` scaling
    exponent reaches ``target_exponent`` (3 = Murray's law), early-stopping at the closest
    approach. Returns the starting vs. final exponent and how far the tree moved toward
    the target — the classic emergent optimum of a self-organising vasculature."""
    from .core.pipeline import run_adaptation

    nx, ny = int(np.clip(nx, 30, 160)), int(np.clip(ny, 30, 160))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    ad = run_adaptation(prm, grid, target_exponent=target_exponent, rate=rate,
                        n_iter=n_iter, domain_limit=_DOMAIN)
    return {"target_exponent": target_exponent, "n_iter": ad.n_iter,
            "murray_start": ad.murray_start, "murray_final": ad.murray_exponent,
            "moved_toward_target": abs(ad.murray_exponent - target_exponent)
            < abs(ad.murray_start - target_exponent)}


def multicompartment(n: int = 3, beta: float = 1.0, nx: int = 90, ny: int = 110,
                     p_in: float = 10.0, p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
    """Multi-compartment Darcy perfusion (arteriole → … → venule, Michler/Hyde).

    Couples the fractal trees to ``n`` stacked porous compartments exchanging at rate
    ``beta``; blood enters compartment 0 and drains compartment n−1. Returns the
    throughflow (total inter-compartment exchange = capillary perfusion) and the mean
    pressure of each compartment, confirming the arteriole→venule pressure cascade."""
    from .core.pipeline import run_multicompartment

    n = int(np.clip(n, 2, 5))
    nx, ny = int(np.clip(nx, 30, 200)), int(np.clip(ny, 30, 200))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY)
    res = run_multicompartment(prm, grid, permeabilities=tuple([1.0] * n),
                               betas=tuple([float(beta)] * (n - 1)), domain_limit=_DOMAIN)
    tf = res.solution.terminal_flux
    return {"n_compartments": n, "beta": beta,
            "arterial_inflow": float(tf[tf > 0].sum()),
            "capillary_perfusion": float(np.nansum(res.perfusion[0])),
            "compartment_mean_pressure": [float(np.nanmean(res.compartments[c])) for c in range(n)]}


def tree_morphometry(kind: str = "artery", r_min_frac: float = 0.15, seed: int = 0) -> dict:
    """Horton–Strahler morphometry of a fractal tree: the bifurcation (R_B), length
    (R_L) and diameter (R_D) ratios plus the max Strahler order and per-order segment
    counts — the standard morphometric fingerprint of a vascular tree."""
    from .core.geometry import arterial_tree, venous_tree
    from .core.morphometry import strahler_summary

    prm = Parameters(r_min_frac=r_min_frac, seed=seed)
    net = venous_tree(prm) if kind == "vein" else arterial_tree(prm)
    s = strahler_summary(net)
    return {"kind": kind, "max_order": s["max_order"], "counts": s["counts"],
            "bifurcation_ratio": s["bifurcation_ratio"], "length_ratio": s["length_ratio"],
            "diameter_ratio": s["diameter_ratio"]}


def prune_tree(kind: str = "artery", steps: int = 1, r_min_frac: float = 0.15,
               seed: int = 0) -> dict:
    """Prune ``steps`` generations of terminal branches off a fractal tree (the
    inverse of growth): each step removes every free-end branch back to its nearest
    junction, never the trunk to the inlet. Reports segments/terminals and the max
    Strahler order before vs after."""
    from .core.geometry import arterial_tree, prune_terminal_branches, venous_tree
    from .core.morphometry import strahler_summary

    prm = Parameters(r_min_frac=r_min_frac, seed=seed)
    net = venous_tree(prm) if kind == "vein" else arterial_tree(prm)
    pruned = prune_terminal_branches(net, int(np.clip(steps, 0, 50)))

    def stats(n):
        return {"segments": int(n.n_segments), "terminals": int(n.terminal_nodes.size),
                "max_order": int(strahler_summary(n)["max_order"]) if n.n_segments else 0}

    return {"kind": kind, "steps": int(steps), "before": stats(net), "after": stats(pruned),
            "removed_segments": int(net.n_segments - pruned.n_segments)}


def generate_trees(dim: int = 2, shape: str = "ellipse", generations: int = 4,
                   fill: str = "colonization", fill_steps: int = 40, n_terminals: int = 60,
                   artery_root=None, vein_root=None, artery_direction=None,
                   vein_direction=None, artery_angle: float = 90.0, vein_angle: float = -90.0,
                   solve: bool = True, seed: int = 0, extent: float = 1.0,
                   _return_model: bool = False) -> dict:
    """Grow an arterial + venous tree into a tissue shape (2-D or 3-D) and solve.

    A fractal skeleton of ``generations`` branching levels starts at each root
    (``*_root`` = [x, y] or [x, y, z]; 2-D trunk angle in degrees, 3-D direction
    vector), then ``fill`` ("none" | "colonization" | "cco") grows it on to fill the
    ``shape`` — any organ in :mod:`perfusion.organs` (2-D: heart, kidney, brain,
    liver, lung, tongue, tumour, placenta, ellipse, …; 3-D: heart, kidney, brain,
    liver, lung, tumour, ellipsoid, box, …). ``extent`` scales the domain (>1 = a
    bigger object). 3-D trees branch in twisting planes (truly volumetric).
    Returns per-tree stats and, with ``solve``, the perfused inflow."""
    from . import organs, treegen

    d3 = int(dim) == 3
    e = float(np.clip(extent, 0.25, 16.0))
    dim_ = 3 if d3 else 2
    # unspecified roots/aims attach where the organ's vessels do anatomically
    anat = organs.roots(shape, dim_, treegen.domain_for(dim_, e))

    def aim(kind, given_dir, given_angle, default_angle):
        a = anat[kind]
        if d3:
            return tuple(given_dir or a["direction"]), 90.0
        return (0, 0, 1), (given_angle if given_angle != default_angle else a["angle"])

    a_dir, a_ang = aim("artery", artery_direction, artery_angle, 90.0)
    v_dir, v_ang = aim("vein", vein_direction, vein_angle, -90.0)
    spec = treegen.GenSpec(
        dim=dim_, shape=shape, generations=int(np.clip(generations, 1, 10)),
        fill=fill, fill_steps=int(np.clip(fill_steps, 0, 400)),
        n_terminals=int(np.clip(n_terminals, 1, 600)), seed=seed, nx=20 if d3 else 60,
        extent=e,
        artery=treegen.TreeSpec(tuple(artery_root or anat["artery"]["root"]), a_ang, a_dir),
        vein=treegen.TreeSpec(tuple(vein_root or anat["vein"]["root"]), v_ang, v_dir,
                              radius=2.4, gamma=1.2))
    art, ven, mask, domain, _ = treegen.build(spec)
    out = {"dim": spec.dim, "shape": shape, "fill": fill, "active_cells": int(mask.sum()),
           "artery": treegen.tree_stats(art), "vein": treegen.tree_stats(ven)}
    if _return_model:                 # internal: tissue_analysis reuses the built model
        out["_model"] = (spec, art, ven, mask, domain)
    if solve:       # the same calculations in 2-D and 3-D: flow, flux, time-of-flight
        res = treegen.solve(spec, art, ven, mask, domain)
        q = res.solution.terminal_flux
        out["inflow"] = float(q[q > 0].sum())
        out["outflow"] = float(-q[q < 0].sum())
        out["homogeneity"] = treegen.homogeneity(res)
    return out


def tissue_analysis(dim: int = 3, shape: str = "kidney", generations: int = 4,
                    fill: str = "cco", n_terminals: int = 40, extent: float = 1.0,
                    k: float = 1.0, adapt: bool = False, n_frames: int = 30,
                    model: str = "1CM", seed: int = 0) -> dict:
    """Grow artery + vein trees into an organ (2-D or 3-D), solve the hybrid model and
    run the downstream analyses on that tissue — the same chain in both dimensions:

    - optional wall-shear **adaptation** of the radii toward Murray's law (``adapt``),
    - **transit-time** metrics (MTT, CTH) of the Darcy time-of-flight,
    - steady **oxygen** extraction (OEF) at Damköhler number ``k``,
    - a contrast-**tracer** bolus advected through the bed (``n_frames`` snapshots) and a
      tracer-kinetic fit (``model``: MS / 1CM / 2CUM / 2CXM) of its tissue curve.
    """
    from . import treegen
    from .core import tkm, transport
    from .core.oxygen import oxygen_from_result
    from .core.parameters import Parameters
    from .core.perfusion_metrics import transit_metrics
    from .core.pipeline import run_adaptation

    grown = generate_trees(dim=dim, shape=shape, generations=generations, fill=fill,
                           n_terminals=n_terminals, solve=False, seed=seed, extent=extent,
                           _return_model=True)
    spec, art, ven, mask, domain = grown["_model"]
    res = treegen.solve(spec, art, ven, mask, domain)
    out = {"dim": spec.dim, "shape": shape, "active_cells": int(mask.sum()),
           "n_terminals": {"artery": int(art.terminal_nodes.size),
                           "vein": int(ven.terminal_nodes.size)}}
    if adapt:
        ad = run_adaptation(Parameters(p_in=10.0, p_out=0.0), res.grid,
                            domain_limit=domain, art=art, ven=ven, n_iter=40)
        res = ad.result
        out["adaptation"] = {"murray_start": ad.murray_start,
                             "murray_final": ad.murray_exponent, "n_iter": ad.n_iter}
    q = res.solution.terminal_flux
    out["inflow"] = float(q[q > 0].sum())
    tm = transit_metrics(np.where(mask, res.tof, np.nan), weights=res.flux)
    out["transit"] = {"mtt": tm.mtt, "cth": tm.cth, "rd": tm.rd}
    ox = oxygen_from_result(res, k=float(k))
    out["oxygen"] = {"k": float(k), "oef": ox.oef}

    finite = res.tof[np.isfinite(res.tof)]
    duration = 5.0 * max(float(np.percentile(finite, 80)) if finite.size else 5.0, 1e-3)
    n_frames = int(np.clip(n_frames, 8, 120))
    n_steps = max(n_frames, 400)
    t = np.arange(n_steps) * duration / n_steps
    aif = tkm.gamma_variate_aif(t, t0=0.08 * duration, A=3.0, B=0.06 * duration)
    aif = aif / aif.max()
    frames = transport.transport_forward(res.grid, res.fx, res.fy, res.sources, q, aif,
                                         t[1] - t[0], theta=1.0, fz=getattr(res, "fz", None))
    stride = max(1, n_steps // n_frames)
    fr, tt, ca = frames[::stride][:n_frames], t[::stride][:n_frames], aif[::stride][:n_frames]
    conc = np.array([float(f[mask].mean()) for f in fr])
    fit = tkm.fit(tt, ca, conc, model=model)

    def num(x):
        return float(x) if x is not None and np.isfinite(x) else None

    out["tracer"] = {"model": model.upper(), "Fp": num(fit.Fp), "vp": num(fit.vp),
                     "PS": num(fit.PS), "ve": num(fit.ve),
                     "peak_time": float(tt[int(np.argmax(conc))]),
                     "peak_concentration": float(conc.max())}
    return out


def optimize_tissue_trees(dim: int = 2, shape: str = "kidney", kind: str = "tof_std",
                          free=None, method: str = "local", maxiter: int = 15,
                          generations: int = 4, fill: str = "cco", n_terminals: int = 40,
                          extent: float = 1.0, seed: int = 0) -> dict:
    """Optimise artery + vein trees for homogeneous perfusion **in a given organ**
    (Paper D, organ-aware; 2-D or 3-D). Grows the organ's trees from its anatomical
    roots with ``fill``, then searches the ``free`` tunables (``a_gamma``, ``v_gamma``,
    ``length_to_radius``, ``exponent``, ``v_radius_scale``, 3-D ``twist``) to minimise
    the homogeneity objective ``kind`` (``tof_std``, ``tof_max``, ``flux_std``,
    ``flux_range``, ``flux_l2``). Returns the before/after objective and best values."""
    from . import treegen

    grown = generate_trees(dim=dim, shape=shape, generations=generations, fill=fill,
                           n_terminals=n_terminals, solve=False, seed=seed, extent=extent,
                           _return_model=True)
    spec = grown["_model"][0]
    r = treegen.optimize(spec, kind=kind, free=free, method=method,
                         maxiter=int(np.clip(maxiter, 1, 60)), seed=seed)
    gain = (r.start - r.best) / r.start if np.isfinite(r.start) and r.start > 0 else 0.0
    return {"dim": spec.dim, "shape": shape, "kind": kind, "start": r.start, "best": r.best,
            "improvement": gain, "n_eval": r.n_eval, "free": r.free,
            "values": {k: float(v) for k, v in r.values.items()}}


def data_constrained_growth(hotspot_weight: float = 8.0, radius_frac: float = 0.22,
                            n_points: int = 500, seed: int = 0) -> dict:
    """Data-constrained vascular growth toward a target perfusion-density map.

    Grows a tree by space colonisation with the attraction points biased toward a
    central demand hotspot (weight ``hotspot_weight`` vs. 1 elsewhere), then reports the
    fraction of grown terminals that landed inside the hotspot vs. the hotspot's area
    fraction — a ratio > 1 means the tree concentrated where demand is high."""
    from . import growth
    from .core.geometry import arterial_tree

    nx, ny = 80, 90
    xc, yc = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rx, ry = radius_frac * nx, radius_frac * ny
    hot = ((xc - nx / 2) / rx) ** 2 + ((yc - ny / 2) / ry) ** 2 <= 1.0
    density = np.where(hot, float(hotspot_weight), 1.0)
    mask = np.ones((nx, ny), bool)

    base = arterial_tree(Parameters(r_min_frac=0.5, seed=seed))
    net = growth.grow_in_mask(base, mask, _DOMAIN, n_points=int(n_points), seed=seed,
                              density=density, step=4.0)
    # map terminal coords back to grid cells to test concentration
    tn = net.terminal_nodes
    x0, x1, y0, y1 = _DOMAIN
    ix = np.clip(((net.node_x[tn] - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((net.node_y[tn] - y0) / (y1 - y0) * ny).astype(int), 0, ny - 1)
    in_hot = float(np.mean(hot[ix, iy])) if tn.size else 0.0
    area = float(hot.mean())
    return {"hotspot_weight": hotspot_weight, "n_terminals": int(tn.size),
            "terminal_fraction_in_hotspot": in_hot, "hotspot_area_fraction": area,
            "concentration_ratio": in_hot / area if area > 0 else None}


def anisotropic_perfusion(anisotropy: float = 5.0, nx: int = 90, ny: int = 110,
                          p_in: float = 10.0, p_out: float = 0.0,
                          r_min_frac: float = 0.15) -> dict:
    """Perfusion on an anisotropic capillary bed (diagonal-tensor permeability).

    Sets ``Ky = anisotropy · Kx`` (``Kx = 1``) and runs the hybrid model, reporting the
    mean Darcy flux resolved along x vs. y — showing how the high-permeability axis
    channels the flow. ``anisotropy = 1`` is isotropic."""
    nx, ny = int(np.clip(nx, 30, 200)), int(np.clip(ny, 30, 200))
    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY, permeability=1.0,
                permeability_y=float(anisotropy))
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN)
    fx, fy = result.fx, result.fy
    return {"anisotropy": anisotropy,
            "mean_abs_flux_x": float(np.abs(fx).mean()),
            "mean_abs_flux_y": float(np.abs(fy).mean()),
            "flux_mean": float(result.flux.mean())}


def heterogeneous_perfusion(lesion_k: float = 0.02, radius_frac: float = 0.2,
                            nx: int = 90, ny: int = 110, p_in: float = 10.0,
                            p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
    """Perfusion with a heterogeneous permeability field (a low-``K`` tissue lesion).

    Places a central elliptical patch of permeability ``lesion_k`` (relative to the
    healthy ``K = 1``) covering ``radius_frac`` of the domain, runs the hybrid model,
    and compares the mean Darcy flux inside vs. outside the lesion — showing how a
    low-permeability region starves local perfusion (ischemia). Uses the harmonic-mean
    transmissibilities so the lesion throttles flow at its boundary faces."""
    nx, ny = int(np.clip(nx, 30, 200)), int(np.clip(ny, 30, 200))
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    rx, ry = radius_frac * nx, radius_frac * ny
    lesion = ((ix - nx / 2) / rx) ** 2 + ((iy - ny / 2) / ry) ** 2 <= 1.0
    kfield = np.ones((nx, ny))
    kfield[lesion] = float(lesion_k)

    prm = Parameters(p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)
    grid = Grid(nx=nx, ny=ny, lx=_LX, ly=_LY, permeability=kfield)
    result = run_hybrid(prm, grid, domain_limit=_DOMAIN)
    flux = result.flux
    inside = float(flux[lesion].mean())
    outside = float(flux[~lesion].mean())
    return {"lesion_k": lesion_k, "flux_in_lesion": inside, "flux_outside": outside,
            "starvation_ratio": inside / outside if outside > 0 else None,
            "flux_mean": float(flux.mean()), "flux_std": float(flux.std())}


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
    def calibrate_parameter_to_flow(target_inflow: float, param: str = "permeability",
                                    lo: float = 0.1, hi: float = 10.0, nx: int = 70,
                                    ny: int = 80, p_in: float = 10.0, p_out: float = 0.0,
                                    r_min_frac: float = 0.15) -> dict:
        """Inverse calibration: recover a model input (permeability / p_in / mu) so the
        model's arterial inflow matches `target_inflow` (bounded 1-D search over forward
        solves). Returns the recovered value + achieved inflow — fit the model to data."""
        return calibrate_to_flow(target_inflow=target_inflow, param=param, lo=lo, hi=hi,
                                 nx=nx, ny=ny, p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)

    @server.tool()
    def uncertainty_quantification(n_samples: int = 24, p_in_low: float = 5.0,
                                   p_in_high: float = 20.0, mu_low: float = 0.8,
                                   mu_high: float = 1.5, rmin_low: float = 0.35,
                                   rmin_high: float = 0.6, nx: int = 60, ny: int = 70,
                                   seed: int = 0) -> dict:
        """Monte-Carlo uncertainty quantification: Latin-hypercube-sample the inlet
        pressure, viscosity and branch depth over the given ranges, re-solve n_samples
        times, and return each quantity of interest's mean/std plus the sensitivity
        (Pearson r) of arterial inflow to each input. A batch of full solves — keep the
        grid coarse and n_samples modest."""
        return uncertainty(n_samples=n_samples, p_in_range=(p_in_low, p_in_high),
                           mu_range=(mu_low, mu_high), rmin_range=(rmin_low, rmin_high),
                           nx=nx, ny=ny, seed=seed)

    @server.tool()
    def sobol_sensitivity_indices(n_samples: int = 48, p_in_low: float = 5.0,
                                  p_in_high: float = 20.0, mu_low: float = 0.8,
                                  mu_high: float = 1.5, rmin_low: float = 0.35,
                                  rmin_high: float = 0.6, nx: int = 60, ny: int = 70,
                                  seed: int = 0) -> dict:
        """Variance-based Sobol sensitivity (polynomial-chaos surrogate): first-order
        (S1) and total-effect (ST) indices of arterial inflow to inlet pressure,
        viscosity and branch depth — the rigorous variance decomposition (ST ≥ S1;
        ST − S1 = interaction). Returns the surrogate fit R² too. Keep the grid coarse."""
        return sobol_sensitivity(n_samples=n_samples, p_in_range=(p_in_low, p_in_high),
                                 mu_range=(mu_low, mu_high), rmin_range=(rmin_low, rmin_high),
                                 nx=nx, ny=ny, seed=seed)

    @server.tool()
    def adapt_vessel_network(target_exponent: float = 3.0, rate: float = 0.05,
                             n_iter: int = 60, nx: int = 70, ny: int = 80, p_in: float = 10.0,
                             p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
        """Structural adaptation (Pries–Secomb): evolve vessel radii under wall-shear
        remodelling until the |Q| ∝ r^k scaling exponent reaches `target_exponent`
        (3 = Murray's law), early-stopping at the closest approach. Returns the start
        vs. final exponent — the emergent optimum of a self-organising vasculature."""
        return adapt_network(target_exponent=target_exponent, rate=rate, n_iter=n_iter,
                             nx=nx, ny=ny, p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)

    @server.tool()
    def multicompartment_perfusion(n: int = 3, beta: float = 1.0, nx: int = 90, ny: int = 110,
                                   p_in: float = 10.0, p_out: float = 0.0,
                                   r_min_frac: float = 0.15) -> dict:
        """Multi-compartment Darcy perfusion (arteriole → … → venule): couple the
        fractal trees to `n` stacked porous compartments exchanging at rate `beta`.
        Blood enters compartment 0 (arterial terminals) and drains compartment n−1
        (venous terminals), percolating through the exchange. Returns the capillary
        perfusion (total exchange) + each compartment's mean pressure. Keep grid coarse."""
        return multicompartment(n=n, beta=beta, nx=nx, ny=ny, p_in=p_in, p_out=p_out,
                                r_min_frac=r_min_frac)

    @server.tool()
    def strahler_morphometry(kind: str = "artery", r_min_frac: float = 0.15,
                             seed: int = 0) -> dict:
        """Horton–Strahler morphometry of a fractal tree: bifurcation R_B, length R_L
        and diameter R_D ratios + max Strahler order and per-order segment counts —
        the standard morphometric fingerprint used to validate a vascular tree."""
        return tree_morphometry(kind=kind, r_min_frac=r_min_frac, seed=seed)

    @server.tool()
    def prune_vessel_tree(kind: str = "artery", steps: int = 1, r_min_frac: float = 0.15,
                          seed: int = 0) -> dict:
        """Prune the outermost `steps` generations of terminal branches off a fractal
        artery/vein tree — the inverse of growth (the trunk to the inlet is kept).
        Returns segment/terminal counts and max Strahler order before vs after."""
        return prune_tree(kind=kind, steps=steps, r_min_frac=r_min_frac, seed=seed)

    @server.tool()
    def grow_trees_in_tissue(dim: int = 2, shape: str = "ellipse", generations: int = 4,
                             fill: str = "colonization", fill_steps: int = 40,
                             n_terminals: int = 60, artery_root: list[float] | None = None,
                             vein_root: list[float] | None = None,
                             artery_direction: list[float] | None = None,
                             vein_direction: list[float] | None = None,
                             artery_angle: float = 90.0, vein_angle: float = -90.0,
                             solve: bool = True, seed: int = 0, extent: float = 1.0) -> dict:
        """Grow artery + vein trees into a tissue shape in 2-D or 3-D: a fractal
        skeleton (`generations` levels) from placed roots, then a space-filling
        `fill` (colonization steps / CCO terminals). Shapes are organs — 2-D: heart,
        kidney, brain, liver, lung, tongue, tumour, placenta, ellipse, rectangle,
        wedge; 3-D: heart, kidney, brain, liver, lung, tumour, ellipsoid, box,
        cylinder, hemisphere. `extent` scales the domain (>1 = bigger object). Returns
        segment/terminal/Strahler stats per tree and the perfused inflow/outflow."""
        return generate_trees(dim=dim, shape=shape, generations=generations, fill=fill,
                              fill_steps=fill_steps, n_terminals=n_terminals,
                              artery_root=artery_root, vein_root=vein_root,
                              artery_direction=artery_direction,
                              vein_direction=vein_direction, artery_angle=artery_angle,
                              vein_angle=vein_angle, solve=solve, seed=seed, extent=extent)

    @server.tool()
    def optimize_trees_in_tissue(dim: int = 2, shape: str = "kidney", kind: str = "tof_std",
                                 free: list[str] | None = None, method: str = "local",
                                 maxiter: int = 15, generations: int = 4, fill: str = "cco",
                                 n_terminals: int = 40, extent: float = 1.0,
                                 seed: int = 0) -> dict:
        """Optimise artery + vein trees for homogeneous perfusion in a chosen organ
        (2-D or 3-D; anatomical roots, space-filling `fill`). Varies `free` from
        a_gamma, v_gamma, length_to_radius, exponent, v_radius_scale, twist (3-D) to
        minimise `kind` (tof_std, tof_max, flux_std, flux_range, flux_l2); `method`
        local (Nelder-Mead) or global (differential evolution). Returns before/after
        objective and the best values."""
        return optimize_tissue_trees(dim=dim, shape=shape, kind=kind, free=free,
                                     method=method, maxiter=maxiter, generations=generations,
                                     fill=fill, n_terminals=n_terminals, extent=extent,
                                     seed=seed)

    @server.tool()
    def tissue_perfusion_analysis(dim: int = 3, shape: str = "kidney", generations: int = 4,
                                  fill: str = "cco", n_terminals: int = 40,
                                  extent: float = 1.0, k: float = 1.0, adapt: bool = False,
                                  n_frames: int = 30, model: str = "1CM",
                                  seed: int = 0) -> dict:
        """Grow artery + vein trees into an organ (2-D or 3-D, see
        grow_trees_in_tissue for shapes), solve, then analyse that tissue: optional
        Murray adaptation of radii (`adapt`), transit-time MTT/CTH, oxygen extraction
        (OEF at Damköhler `k`), and a contrast-tracer bolus with a tracer-kinetic fit
        (`model` MS/1CM/2CUM/2CXM → Fp, vp, PS, ve). Works identically in 3-D."""
        return tissue_analysis(dim=dim, shape=shape, generations=generations, fill=fill,
                               n_terminals=n_terminals, extent=extent, k=k, adapt=adapt,
                               n_frames=n_frames, model=model, seed=seed)

    @server.tool()
    def data_constrained_tree_growth(hotspot_weight: float = 8.0, radius_frac: float = 0.22,
                                     n_points: int = 500, seed: int = 0) -> dict:
        """Data-constrained vascular growth: grow a tree with attraction points biased
        toward a central demand hotspot (weight vs 1 elsewhere) and report the fraction
        of terminals inside the hotspot vs its area — concentration_ratio > 1 means the
        tree grew denser where demand is high (matching a target perfusion map)."""
        return data_constrained_growth(hotspot_weight=hotspot_weight, radius_frac=radius_frac,
                                       n_points=n_points, seed=seed)

    @server.tool()
    def anisotropic_perfusion_bed(anisotropy: float = 5.0, nx: int = 90, ny: int = 110,
                                  p_in: float = 10.0, p_out: float = 0.0,
                                  r_min_frac: float = 0.15) -> dict:
        """Perfusion on an anisotropic (diagonal-tensor) capillary bed: set Ky =
        anisotropy·Kx (Kx=1) and report the mean Darcy flux along x vs. y — the
        high-permeability axis channels the flow. anisotropy=1 is isotropic."""
        return anisotropic_perfusion(anisotropy=anisotropy, nx=nx, ny=ny, p_in=p_in,
                                     p_out=p_out, r_min_frac=r_min_frac)

    @server.tool()
    def heterogeneous_perfusion_lesion(lesion_k: float = 0.02, radius_frac: float = 0.2,
                                       nx: int = 90, ny: int = 110, p_in: float = 10.0,
                                       p_out: float = 0.0, r_min_frac: float = 0.15) -> dict:
        """Perfusion with a heterogeneous permeability field: place a central low-`K`
        tissue lesion (permeability `lesion_k` relative to healthy K=1, covering
        `radius_frac` of the domain) and compare the mean Darcy flux inside vs.
        outside — showing how low permeability starves local perfusion (ischemia).
        Uses harmonic-mean transmissibilities. A full hybrid solve — keep grid coarse."""
        return heterogeneous_perfusion(lesion_k=lesion_k, radius_frac=radius_frac, nx=nx,
                                       ny=ny, p_in=p_in, p_out=p_out, r_min_frac=r_min_frac)

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
