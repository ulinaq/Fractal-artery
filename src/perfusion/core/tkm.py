"""Tracer-kinetic models (TKMs) for DCE-MRI perfusion quantification.

Implements the four models compared in Paper C (thesis Chapter 3): the maximum
slope (MS) model, one-compartment model (1CM), two-compartment exchange model
(2CXM) and two-compartment uptake model (2CUM). Each maps an arterial input
function (AIF) to a tissue concentration curve; fitting recovers plasma flow
(Fp) and volume (vp).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt


def _convolve(a: np.ndarray, b: np.ndarray, dt: float) -> np.ndarray:
    """Discrete causal convolution (a ⊛ b) sampled on a uniform grid of step dt."""
    full = np.convolve(a, b) * dt
    return full[: len(a)]


@dataclass
class TKMResult:
    Fp: float                       # plasma flow [ml/min/100ml]
    vp: float                       # plasma volume [ml/100ml]
    PS: float | None = None         # permeability-surface area product
    ve: float | None = None         # interstitial volume
    curve: np.ndarray | None = None  # fitted tissue concentration


def one_compartment(t: np.ndarray, ca: np.ndarray, Fp: float, vp: float) -> np.ndarray:
    """1CM tissue concentration: C = Fp e^{-t Fp/vp} ⊛ ca (Eq. 3.16)."""
    dt = t[1] - t[0]
    imp = Fp * np.exp(-t * Fp / max(vp, 1e-9))
    return _convolve(imp, ca, dt)


def two_compartment_uptake(t, ca, Fp, vp, PS) -> np.ndarray:
    """2CUM tissue concentration (Eq. 3.31)."""
    dt = t[1] - t[0]
    Tp = vp / max(Fp + PS, 1e-9)
    EF = PS / max(Fp + PS, 1e-9)
    imp = Fp * ((1.0 - EF) * np.exp(-t / max(Tp, 1e-9)) + EF)
    return _convolve(imp, ca, dt)


def two_compartment_exchange(t, ca, Fp, vp, PS, ve) -> np.ndarray:
    """2CXM tissue concentration (Eqs. 3.23-3.25)."""
    dt = t[1] - t[0]
    vp = max(vp, 1e-9)
    ve = max(ve, 1e-9)
    a = (PS + Fp) / vp + PS / ve
    b = 4.0 * PS / ve * Fp / vp
    disc = np.sqrt(max(a * a - b, 0.0))
    x1 = -0.5 * (a - disc)
    x2 = -0.5 * (a + disc)
    denom = x1 - x2 if abs(x1 - x2) > 1e-12 else 1e-12
    A = (x1 + (PS / vp + PS / ve)) / denom
    imp = Fp * (A * np.exp(x1 * t) + (1.0 - A) * np.exp(x2 * t))
    return _convolve(imp, ca, dt)


def maximum_slope(t: np.ndarray, ca: np.ndarray, C: np.ndarray) -> TKMResult:
    """Maximum-slope estimate of plasma flow (Eqs. 3.18-3.19).

    Fp = max_t C'(t) / max_t ca(t); no fitting, so robust but biased for short
    transit times.
    """
    dCdt = np.gradient(C, t)
    Fp = float(np.max(dCdt) / max(np.max(ca), 1e-9))
    return TKMResult(Fp=Fp, vp=float("nan"), curve=None)


def _fit(model, t, ca, C, p0, bounds):
    def resid(p):
        return model(t, ca, *p) - C
    res = opt.least_squares(resid, p0, bounds=bounds, method="trf", max_nfev=10000)
    return res.x, model(t, ca, *res.x)


def fit(t: np.ndarray, ca: np.ndarray, C: np.ndarray, model: str = "1CM") -> TKMResult:
    """Fit a named TKM to a tissue concentration curve.

    Parameters
    ----------
    t : time samples (uniform).
    ca : arterial input function.
    C : measured/simulated tissue concentration.
    model : one of ``"MS"``, ``"1CM"``, ``"2CUM"``, ``"2CXM"``.
    """
    model = model.upper()
    if model == "MS":
        return maximum_slope(t, ca, C)
    if model == "1CM":
        p, curve = _fit(one_compartment, t, ca, C, [1.0, 0.1],
                        ([0, 0], [np.inf, np.inf]))
        return TKMResult(Fp=p[0], vp=p[1], curve=curve)
    if model == "2CUM":
        p, curve = _fit(two_compartment_uptake, t, ca, C, [1.0, 0.1, 0.1],
                        ([0, 0, 0], [np.inf, np.inf, np.inf]))
        return TKMResult(Fp=p[0], vp=p[1], PS=p[2], curve=curve)
    if model == "2CXM":
        p, curve = _fit(two_compartment_exchange, t, ca, C, [1.0, 0.1, 0.1, 0.2],
                        ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
        return TKMResult(Fp=p[0], vp=p[1], PS=p[2], ve=p[3], curve=curve)
    raise ValueError(f"unknown model {model!r}; use MS/1CM/2CUM/2CXM")


def gamma_variate_aif(t: np.ndarray, t0: float = 0.1, A: float = 3.0,
                      B: float = 1.0, C0: float = 1.0) -> np.ndarray:
    """Gamma-variate arterial input function (Eq. 11 of Paper C)."""
    tt = np.clip(t - t0, 0.0, None)
    return C0 * tt**A * np.exp(-tt / B)
