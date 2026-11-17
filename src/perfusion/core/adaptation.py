"""Structural adaptation of vessel radii toward a wall-shear-stress set-point.

A reduced Pries–Secomb model: each segment's radius grows where the wall shear
stress ``τ_w = 4 μ |Q| / (π r³)`` exceeds a target ``τ_ref`` and shrinks where it
falls below, with a metabolic floor that keeps low-flow vessels from vanishing.
Iterated with a flow solve in the loop (see ``pipeline.run_adaptation``) the tree
self-organises so the wall shear is more uniform — driving the network toward
Murray's law ``Q ∝ r³`` (constant shear), the classic emergent optimum.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def wall_shear_stress(radius: np.ndarray, flow: np.ndarray, viscosity: float = 1.0) -> np.ndarray:
    """Poiseuille wall shear stress ``τ_w = 4 μ |Q| / (π r³)`` per segment."""
    r = np.maximum(np.asarray(radius, dtype=float), _EPS)
    return 4.0 * viscosity * np.abs(np.asarray(flow, dtype=float)) / (np.pi * r**3)


def adaptation_signal(radius: np.ndarray, flow: np.ndarray, tau_ref: float | None = None,
                      viscosity: float = 1.0, k_metabolic: float = 0.0,
                      k_shrink: float = 0.0) -> np.ndarray:
    """Dimensionless growth stimulus ``S`` per segment (Pries–Secomb-style).

    ``S = log10(τ_w / τ_ref) + k_metabolic·log10(1 + Q_ref/|Q|) − k_shrink``: positive
    where the vessel is over-sheared (should dilate), negative where it is quiescent.
    ``τ_ref`` is the shear set-point; ``None`` self-scales it to the current mean wall
    shear, so the tree is driven toward *uniform* shear (Murray's law) rather than an
    arbitrary absolute value.
    """
    tau = wall_shear_stress(radius, flow, viscosity)
    pos = tau[tau > 0]
    if tau_ref is None:
        tau_ref = float(pos.mean()) if pos.size else 1.0
    tau_ref = max(tau_ref, _EPS)
    shear = np.log10((tau + 1e-3 * tau_ref) / tau_ref)
    q = np.abs(np.asarray(flow, dtype=float))
    q_ref = np.mean(q[q > 0]) if np.any(q > 0) else 1.0
    metabolic = k_metabolic * np.log10(1.0 + q_ref / (q + _EPS)) if k_metabolic else 0.0
    return shear + metabolic - k_shrink


def adapt_radii(radius: np.ndarray, flow: np.ndarray, tau_ref: float | None = None,
                viscosity: float = 1.0, rate: float = 0.1, r_min=1e-3, r_max=1e3,
                **signal_kw) -> tuple[np.ndarray, np.ndarray]:
    """One structural-adaptation step: ``r ← r · exp(rate·S)``, clamped.

    Returns ``(new_radius, signal)``. The exponential update keeps radii positive;
    ``rate`` is the per-iteration step and ``r_min``/``r_max`` (scalars or per-segment
    arrays) bound the excursion.
    """
    s = adaptation_signal(radius, flow, tau_ref, viscosity, **signal_kw)
    r_new = np.asarray(radius, dtype=float) * np.exp(rate * np.clip(s, -5.0, 5.0))
    return np.clip(r_new, r_min, r_max), s


def murray_exponent(radius: np.ndarray, flow: np.ndarray) -> float:
    """Best-fit exponent ``k`` in ``|Q| ∝ r^k`` (Murray's law gives ``k = 3``).

    A least-squares slope of ``log|Q|`` vs. ``log r`` over flow-carrying segments;
    ``nan`` if fewer than two usable segments.
    """
    r = np.asarray(radius, dtype=float)
    q = np.abs(np.asarray(flow, dtype=float))
    ok = (r > 0) & (q > 0)
    if ok.sum() < 2:
        return float("nan")
    x = np.log(r[ok])
    a = np.column_stack([x, np.ones_like(x)])            # least-squares slope (log Q vs log r)
    slope, _ = np.linalg.lstsq(a, np.log(q[ok]), rcond=None)[0]
    return float(slope)
