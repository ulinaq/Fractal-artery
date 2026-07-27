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


def kinetic_maps(frames: np.ndarray, times: np.ndarray, aif: np.ndarray,
                 mask: np.ndarray | None = None) -> dict:
    """Model-free voxel-wise perfusion maps from a tracer time series.

    ``frames`` is ``(n_t, nx, ny)`` tissue concentration. Returns per-voxel:

    - ``Fp`` — plasma flow by the **maximum-slope** method,
      ``max_t dC/dt / max(ca)`` (Eqs. 3.18-3.19): fast and fit-free.
    - ``vp`` — blood volume as the area-under-curve ratio ``∫C / ∫ca`` (the
      tracer's distribution volume for an intravascular agent).

    Both are ``(nx, ny)``; inactive voxels (``mask`` False) are NaN.
    """
    frames = np.asarray(frames, dtype=float)
    times = np.asarray(times, dtype=float)
    aif = np.asarray(aif, dtype=float)
    dcdt = np.gradient(frames, times, axis=0)
    fp = dcdt.max(axis=0) / max(float(aif.max()), 1e-12)
    aif_auc = float(np.trapezoid(aif, times)) or 1e-12
    vp = np.trapezoid(frames, times, axis=0) / aif_auc
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        fp = np.where(m, fp, np.nan)
        vp = np.where(m, vp, np.nan)
    return {"Fp": fp, "vp": vp}


def deconvolve(times: np.ndarray, ca: np.ndarray, curves: np.ndarray,
               oi: float = 0.1) -> np.ndarray:
    """Truncated-SVD deconvolution of the residue function (Østergaard 1996).

    Solves ``C = A k`` for ``k = Fp·R`` (flow-scaled residue), where ``A`` is the
    lower-triangular convolution matrix of the AIF ``ca`` on a uniform grid.
    Singular values below ``oi · max(s)`` are truncated (Tikhonov-like
    regularisation) to stabilise the ill-posed inverse. One SVD of ``A`` is reused
    across every column of ``curves`` (shape ``(n_t,)`` or ``(n_t, n_curves)``),
    so a whole voxel field deconvolves in a single factorisation.
    """
    times = np.asarray(times, dtype=float)
    ca = np.asarray(ca, dtype=float)
    curves = np.asarray(curves, dtype=float)
    n = len(times)
    dt = float(np.mean(np.diff(times))) if n > 1 else 1.0
    # Lower-triangular Toeplitz convolution matrix A[i, j] = dt · ca[i-j] (i >= j).
    a = np.zeros((n, n))
    idx = np.arange(n)
    lower = idx[:, None] - idx[None, :]
    valid = lower >= 0
    a[valid] = ca[lower[valid]] * dt
    u, s, vt = np.linalg.svd(a)
    s_inv = np.where(s >= oi * (s.max() if s.size else 0.0), 1.0 / np.where(s > 0, s, 1.0), 0.0)
    # k = V diag(s_inv) Uᵀ C, reusing the factorisation for all columns.
    k = vt.T @ (s_inv[:, None] * (u.T @ np.atleast_2d(curves.T).T))
    return k


def deconvolution_maps(frames: np.ndarray, times: np.ndarray, aif: np.ndarray,
                       mask: np.ndarray | None = None, oi: float = 0.1) -> dict:
    """Model-free perfusion maps by residue-function **deconvolution**.

    Unlike :func:`kinetic_maps` (max-slope / AUC heuristics), this recovers the
    flow-scaled residue ``k = Fp·R`` per voxel by truncated-SVD deconvolution and
    reads off, via the central-volume theorem:

    - ``Fp`` — plasma flow, ``max_t k`` (the residue is 1 at ``t=0``),
    - ``CBV`` — blood volume, ``∫k dt`` (== ``Fp·MTT``),
    - ``MTT`` — mean transit time, ``CBV / Fp``.

    All ``(nx, ny)``; inactive voxels (``mask`` False) are NaN.
    """
    frames = np.asarray(frames, dtype=float)
    times = np.asarray(times, dtype=float)
    aif = np.asarray(aif, dtype=float)
    n_t, nx, ny = frames.shape
    flat = frames.reshape(n_t, nx * ny)
    m = np.ones(nx * ny, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    fp = np.full(nx * ny, np.nan)
    cbv = np.full(nx * ny, np.nan)
    mtt = np.full(nx * ny, np.nan)
    if m.any():
        k = deconvolve(times, aif, flat[:, m], oi=oi)         # (n_t, n_active)
        fp_a = k.max(axis=0)
        cbv_a = np.trapezoid(k, times, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mtt_a = np.where(fp_a > 0, cbv_a / fp_a, np.nan)
        fp[m], cbv[m], mtt[m] = fp_a, cbv_a, mtt_a
    return {"Fp": fp.reshape(nx, ny), "CBV": cbv.reshape(nx, ny), "MTT": mtt.reshape(nx, ny)}
