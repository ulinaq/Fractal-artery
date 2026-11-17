"""Transient (pulsatile) perfusion via a lumped Windkessel reduction.

The steady hybrid solve gives a total vascular resistance R; vessel-wall
elasticity (``Parameters.compliance``) gives a total compliance C. Together they
form a compliant reservoir — a **Windkessel**: driven by a pulsatile inlet
pressure p_in(t), the compliance buffers the pulse so tissue perfusion is
smoother than the inflow (the classic Windkessel damping). This 0-D reduction is
the first step toward transient simulation; it reuses the steady solver for R and
the vessel geometry for C, and integrates a small ODE in time.

Pure NumPy — lives in ``core``.
"""

from __future__ import annotations

import numpy as np


def pulsatile_inlet(p_mean: float, pulse_amp: float, period: float, dt: float,
                    n_cycles: int = 4):
    """A simple pulsatile inlet pressure p_in(t) = p_mean + pulse_amp·sin(2πt/T)
    sampled at step ``dt`` over ``n_cycles`` cardiac cycles. Returns ``(t, p_in)``."""
    n = max(1, round(n_cycles * period / dt))
    t = np.arange(n) * dt
    return t, p_mean + pulse_amp * np.sin(2.0 * np.pi * t / period)


def windkessel(p_in: np.ndarray, r_total: float, compliance_total: float, dt: float,
               p_out: float = 0.0, proximal_fraction: float = 0.1) -> dict:
    """Integrate the 2-element Windkessel driven by ``p_in(t)``.

    A compliant arterial compartment at pressure P(t) is fed from the source at
    p_in through a proximal resistance ``Rp = proximal_fraction·R`` and drains to
    p_out through the distal (microvascular) resistance ``Rd = (1-…)·R``:

        C dP/dt = (p_in − P)/Rp − (P − p_out)/Rd,   q_tissue = (P − p_out)/Rd.

    Solved by implicit Euler (unconditionally stable; C = 0 gives the quasi-static
    limit q_tissue = ΔP/R). Returns arrays ``P``, ``q_in``, ``q_tissue``.
    """
    rp = max(proximal_fraction * r_total, 1e-12)
    rd = max((1.0 - proximal_fraction) * r_total, 1e-12)
    c = max(compliance_total, 0.0)
    p_in = np.asarray(p_in, dtype=float)
    P = np.empty_like(p_in)
    prev = float(p_in[0]) if p_in.size else p_out
    denom = c / dt + 1.0 / rp + 1.0 / rd
    for i, pin in enumerate(p_in):
        prev = (c / dt * prev + pin / rp + p_out / rd) / denom
        P[i] = prev
    q_in = (p_in - P) / rp
    q_tissue = (P - p_out) / rd
    return {"P": P, "q_in": q_in, "q_tissue": q_tissue}


def compliance_total(networks, compliance: float) -> float:
    """Lumped vascular compliance C = dV/dp for elastic vessels.

    With ``r = r0(1 + compliance·(p − p_out))`` the segment volume ``π r² L`` gives
    ``dV/dp = 2π·compliance·r0²·L``; summing over all segments of the given
    networks yields the total compliance.
    """
    total = 0.0
    for net in networks:
        if net is None:
            continue
        r, length = np.asarray(net.seg_radius, float), np.asarray(net.seg_length, float)
        total += float(2.0 * np.pi * compliance * np.sum(r ** 2 * length))
    return total
