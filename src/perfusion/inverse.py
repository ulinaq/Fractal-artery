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
from .core.pipeline import run_hybrid

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
              max_iter: int = 40) -> CalibrationResult:
    """Recover ``param`` so the model's ``qoi`` matches ``target`` (1-D bounded search).

    ``param`` is a :class:`Parameters` field (``p_in``/``p_out``/``mu``/``r_min_frac``)
    or ``"permeability"`` (varied on the grid). ``bounds`` brackets the search. Returns
    the recovered value, the achieved QoI and the number of forward solves.
    """
    base = dict(base or {})
    n = {"i": 0}

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
        result = run_hybrid(Parameters(**kw), g, domain_limit=domain_limit, art=art, ven=ven)
        return _quantity(result, qoi)

    res = minimize_scalar(lambda x: (evaluate(x) - target) ** 2, bounds=bounds,
                          method="bounded", options={"maxiter": max_iter})
    return CalibrationResult(param=param, value=float(res.x), target=float(target),
                             achieved=float(evaluate(res.x)), qoi=qoi, n_eval=n["i"])
