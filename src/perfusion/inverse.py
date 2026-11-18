"""Inverse problems / calibration for the perfusion model.

Recover an unknown model input (inlet pressure, viscosity or capillary
permeability) from a *measured* scalar perfusion quantity — e.g. calibrate the
capillary permeability so the simulated arterial inflow matches a measured flow.
A 1-D bounded search over full forward solves (`run_hybrid`); a service layer on the
pure engine, like ``optimize`` / ``uq``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize_scalar

from .core.darcy import Grid
from .core.geometry import Network
from .core.parameters import Parameters

_PARAM_FIELDS = ("p_in", "p_out", "mu", "r_min_frac")


def _quantity(result, qoi: str) -> float:
    tf = result.solution.terminal_flux
    if qoi == "arterial_inflow":
        return float(tf[tf > 0].sum())
    if qoi == "flux_mean":
        flux = result.flux[np.isfinite(result.flux)]
        return float(flux.mean()) if flux.size else 0.0
    if qoi == "tof_mean":
        tof = result.tof[np.isfinite(result.tof)]
        return float(tof.mean()) if tof.size else 0.0
    raise ValueError(f"unknown qoi {qoi!r}")


@dataclass
class CalibrationResult:
    param: str
    value: float          # recovered parameter value
    target: float         # requested QoI value
    achieved: float       # QoI at the recovered value
    qoi: str
    n_eval: int


def calibrate(target: float, param: str, grid: Grid, bounds: tuple[float, float],
              qoi: str = "arterial_inflow", base: dict | None = None,
              domain_limit: tuple[float, float, float, float] = (-70, 70, 0, 160),
              art: Network | None = None, ven: Network | None = None,
              max_iter: int = 40, trees=None) -> CalibrationResult:
    """Recover ``param`` so the model's ``qoi`` matches ``target`` (1-D bounded search).

    ``param`` is a :class:`Parameters` field (``p_in``/``p_out``/``mu``/``r_min_frac``)
    or ``"permeability"`` (varied on the grid). ``bounds`` brackets the search. Returns
    the recovered value, the achieved QoI and the number of forward solves.
    ``trees(prm) -> (art, ven)`` optionally rebuilds the vasculature per evaluation.
    """
    from .uq import _sample_solver

    base = dict(base or {})
    n = {"i": 0}
    solver = _sample_solver([param], grid, domain_limit, art, ven, trees)

    def evaluate(x: float) -> float:
        n["i"] += 1
        kw = dict(base)
        g = grid
        if param == "permeability":
            g = replace(grid, permeability=float(x))
        elif param in _PARAM_FIELDS:
            kw[param] = float(x)
        else:
            raise ValueError(f"unknown param {param!r}")
        # reuses the Darcy factorisation unless param moves terminals / changes K
        result = solver(Parameters(**kw), g)
        return _quantity(result, qoi)

    res = minimize_scalar(lambda x: (evaluate(x) - target) ** 2, bounds=bounds,
                          method="bounded", options={"maxiter": max_iter})
    return CalibrationResult(param=param, value=float(res.x), target=float(target),
                             achieved=float(evaluate(res.x)), qoi=qoi, n_eval=n["i"])


# --------------------------------------------------------------------------- #
# Multi-parameter fit of the hybrid model to a DCE-MRI tissue curve
# --------------------------------------------------------------------------- #
# Inputs a DCE fit may estimate. ``porosity`` is the capillary bed's blood-volume
# fraction: it scales the transit volume (and the measured tissue concentration,
# C_tissue = porosity · c) without changing the flow, so it pairs well with ``p_in``
# (which sets the flow, like Fp) — the hybrid model's (Fp, vp).
DCE_PARAMS = ("p_in", "mu", "permeability", "porosity")


def simulate_tissue_curve(result, times, aif, porosity: float | None = None,
                          n_steps: int | None = None) -> np.ndarray:
    """The mean tissue concentration curve ``C(t) = porosity · mean_active(c)`` of a
    solved hybrid result (2-D or 3-D) for an arterial input ``aif`` sampled at
    ``times`` (non-uniform allowed). The bolus is advected on a fine uniform step
    (implicit Euler) and the curve sampled back at ``times``."""
    from .core import transport

    times = np.asarray(times, dtype=float)
    aif = np.asarray(aif, dtype=float)
    grid = result.grid if porosity is None else replace(result.grid, porosity=float(porosity))
    n = int(n_steps or max(400, 4 * len(times)))
    t_fine = np.linspace(times[0], times[-1], n)
    dt = t_fine[1] - t_fine[0]
    ca = np.interp(t_fine, times, aif)
    frames = transport.transport_forward(
        grid, result.fx, result.fy, result.sources, result.solution.terminal_flux, ca, dt,
        theta=1.0, fz=getattr(result, "fz", None))
    mask = np.asarray(grid.mask, dtype=bool)
    curve = grid.porosity * np.array([float(np.nanmean(f[mask])) for f in frames])
    return np.interp(times, t_fine, curve)


@dataclass
class DCEFitResult:
    params: dict          # fitted value per parameter
    std: dict             # 1-sigma standard error per parameter (None if not identifiable)
    correlation: list     # parameter correlation matrix (rows/cols in ``names`` order)
    names: list
    r2: float             # coefficient of determination of the fit
    rmse: float
    model: list           # the fitted model curve at ``times``
    n_eval: int           # forward simulations (flow solves are reused when possible)
    n_solve: int          # hybrid flow solves
    success: bool
    message: str
    warnings: list


def fit_dce(times, tissue, aif, grid: Grid, params: dict, *, base: dict | None = None,
            start: dict | None = None,
            domain_limit: tuple = (-70, 70, 0, 160), art: Network | None = None,
            ven: Network | None = None, trees=None, max_nfev: int = 80) -> DCEFitResult:
    """Fit several hybrid-model inputs at once to a measured DCE-MRI tissue curve.

    ``times``, ``tissue`` and ``aif`` are the measured samples (tissue + arterial input
    concentration at each time). ``params`` maps each input to estimate (``p_in`` /
    ``mu`` / ``permeability`` / ``porosity``) to its ``(lo, hi)`` bounds; ``base`` holds
    the other :class:`Parameters` fields, ``start`` optional initial guesses (default
    the base value clipped into the bounds, else the bounds' geometric mean).

    Bounded nonlinear least squares in log-parameters (``scipy.optimize.least_squares``,
    trust-region reflective) over full forward simulations: hybrid solve → bolus
    transport → mean tissue curve. Flow solves are cached, so a step that moves only
    ``porosity`` costs one transport, not a solve. Standard errors and the correlation
    matrix come from the Jacobian at the optimum (``s² (JᵀJ)⁻¹``, delta method); a
    near-±1 correlation flags a pair the curve cannot tell apart (e.g. ``p_in`` and
    ``mu``, which only enter the linear model as their ratio).
    """
    from scipy.optimize import least_squares

    from .uq import _sample_solver

    names = list(params)
    if not names:
        raise ValueError("give at least one parameter to fit")
    bad = [n for n in names if n not in DCE_PARAMS]
    if bad:
        raise ValueError(f"cannot fit {bad}; choose from {list(DCE_PARAMS)}")
    times = np.asarray(times, dtype=float)
    tissue = np.asarray(tissue, dtype=float)
    aif = np.asarray(aif, dtype=float)
    if not (times.shape == tissue.shape == aif.shape) or times.ndim != 1:
        raise ValueError("times, tissue and aif must be 1-D and the same length")
    if len(times) <= len(names):
        raise ValueError("need more samples than fitted parameters")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    lo = np.array([float(params[n][0]) for n in names])
    hi = np.array([float(params[n][1]) for n in names])
    if np.any(lo <= 0) or np.any(hi <= lo):
        raise ValueError("bounds must be positive with lo < hi")
    base = dict(base or {})
    start = dict(start or {})

    def initial(i, n):
        if n in start:
            return float(start[n])
        v = base.get(n, grid.permeability if n == "permeability" else None)   # porosity:
        if np.ndim(v):                                                     # mid-range
            v = None
        return float(v) if v is not None else float(np.sqrt(lo[i] * hi[i]))

    x0 = np.log(np.clip([initial(i, n) for i, n in enumerate(names)], lo * 1.0001, hi / 1.0001))
    flow_names = [n for n in names if n != "porosity"]
    solver = _sample_solver(flow_names, grid, domain_limit, art, ven, trees)
    flows: dict = {}
    count = {"eval": 0}

    def model(logx):
        count["eval"] += 1
        vals = dict(zip(names, np.exp(logx), strict=True))
        key = tuple(round(float(vals[n]), 12) for n in flow_names)
        if key not in flows:
            kw = {**base, **{n: vals[n] for n in flow_names if n not in ("permeability",)}}
            g = grid if "permeability" not in vals else replace(grid,
                                                                permeability=vals["permeability"])
            flows[key] = solver(Parameters(**kw), g)
        return simulate_tissue_curve(flows[key], times, aif, porosity=vals.get("porosity"))

    scale = float(np.max(np.abs(tissue))) or 1.0

    def residual(logx):
        return (model(logx) - tissue) / scale

    sol = least_squares(residual, x0, bounds=(np.log(lo), np.log(hi)), method="trf",
                        diff_step=1e-4, x_scale=1.0, max_nfev=max_nfev)
    fitted = np.exp(sol.x)
    curve = residual(sol.x) * scale + tissue
    ssr = float(np.sum((curve - tissue) ** 2))
    sst = float(np.sum((tissue - tissue.mean()) ** 2)) or 1.0
    dof = max(1, len(times) - len(names))
    warnings = []
    J = sol.jac * scale                           # d(curve)/d(log x)
    # a parameter held at a bound has its Jacobian column zeroed by the solver: the
    # error bars come from the free ones (a bound value has no standard error)
    free = np.flatnonzero(sol.active_mask == 0)
    corr = np.full((len(names), len(names)), np.nan)
    std = {n: None for n in names}
    if free.size:
        Jf = J[:, free]
        try:
            cov_log = (ssr / dof) * np.linalg.inv(Jf.T @ Jf)
            sd_log = np.sqrt(np.clip(np.diag(cov_log), 0.0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr[np.ix_(free, free)] = cov_log / np.outer(sd_log, sd_log)
            for k, i in enumerate(free):
                std[names[i]] = float(fitted[i] * sd_log[k])
        except np.linalg.LinAlgError:
            warnings.append("the parameters are not identifiable from this curve "
                            "(singular Jacobian)")
    warnings.extend(
        f"{names[i]} and {names[j]} are strongly correlated ({corr[i, j]:+.2f}): the "
        "curve hardly tells them apart"
        for i in range(len(names)) for j in range(i + 1, len(names))
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.95)
    at_bound = [n for i, n in enumerate(names)
                if sol.active_mask[i] != 0 or np.isclose(fitted[i], lo[i], rtol=1e-3)
                or np.isclose(fitted[i], hi[i], rtol=1e-3)]
    if at_bound:
        warnings.append(f"stopped at a search bound: {', '.join(at_bound)} — widen it")
    return DCEFitResult(
        params={n: float(fitted[i]) for i, n in enumerate(names)}, std=std,
        correlation=np.where(np.isfinite(corr), np.round(corr, 4), None).tolist(),
        names=names, r2=1.0 - ssr / sst, rmse=float(np.sqrt(ssr / len(times))),
        model=curve.tolist(), n_eval=count["eval"], n_solve=len(flows),
        success=bool(sol.success), message=str(sol.message), warnings=warnings)
