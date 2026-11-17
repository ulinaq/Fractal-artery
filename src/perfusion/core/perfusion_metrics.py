"""Perfusion summary metrics derived from the time-of-flight (TOF) field.

The distribution of transit times across the perfused tissue is the *transit-time
distribution* h(τ). From it come the classic clinical-perfusion scalars:

- **MTT** — mean transit time.
- **CTH** — capillary transit-time heterogeneity: the standard deviation of the
  transit times (Jespersen & Østergaard, 2012).
- **RD** — relative dispersion, ``CTH / MTT`` (dimensionless).

A key, non-obvious consequence: for the *same* MTT, a more heterogeneous flow
(larger CTH) extracts *less* oxygen — fast-transit paths shunt blood through
before it can give up its oxygen. We quantify that with a Kety single-capillary
extraction model integrated over the empirical distribution
(``oxygen_extraction_fraction``); because the single-capillary extraction
``1 - exp(-k τ)`` is concave, Jensen's inequality gives exactly this behaviour.

Pure NumPy — lives in ``core`` and depends only on the TOF field a solve already
produces (``HybridResult.tof``). Weighting each cell by its local flux magnitude
gives the flow-weighted distribution (the physically meaningful one); the default
is unweighted (per active cell).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransitMetrics:
    """Scalar summary of a transit-time distribution."""

    mtt: float          # mean transit time
    cth: float          # capillary transit-time heterogeneity (std. dev.)
    rd: float           # relative dispersion = cth / mtt
    n: int              # number of contributing cells

    def as_dict(self) -> dict:
        return {"mtt": self.mtt, "cth": self.cth, "rd": self.rd, "n": self.n}


def _flatten_finite(tof, weights=None):
    """Flatten to finite transit times (and aligned, non-negative weights)."""
    t = np.asarray(tof, dtype=float).reshape(-1)
    ok = np.isfinite(t)
    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        ok &= np.isfinite(w) & (w > 0)
        w = w[ok]
    return t[ok], w


def transit_metrics(tof, weights=None) -> TransitMetrics:
    """MTT / CTH / RD of the transit-time distribution in ``tof``.

    ``tof`` may hold NaNs (masked / unreached cells); they are dropped. Pass a
    flux field as ``weights`` for the flow-weighted distribution.
    """
    t, w = _flatten_finite(tof, weights)
    if t.size == 0:
        return TransitMetrics(0.0, 0.0, 0.0, 0)
    mtt = float(np.average(t, weights=w))
    var = float(np.average((t - mtt) ** 2, weights=w))
    cth = float(np.sqrt(max(var, 0.0)))
    rd = cth / mtt if mtt > 0 else 0.0
    return TransitMetrics(mtt, cth, rd, int(t.size))


def transit_histogram(tof, weights=None, bins: int = 40):
    """Normalised transit-time distribution h(τ): ``(centres, density)``.

    ``density`` integrates to 1 over τ (a probability density), so it is directly
    comparable across simulations with different cell counts.
    """
    t, w = _flatten_finite(tof, weights)
    if t.size == 0:
        return np.zeros(bins), np.zeros(bins)
    counts, edges = np.histogram(t, bins=bins, weights=w, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def oxygen_extraction_fraction(tof, weights=None, k: float = 1.0) -> float:
    """Mean oxygen-extraction fraction over the transit-time distribution.

    Uses the Kety single-capillary extraction ``Q(τ) = 1 - exp(-k τ)`` and
    averages it over h(τ): ``OEF = <1 - exp(-k τ)>``. ``k`` is an effective
    permeability-surface-area rate (per unit transit time). Because ``Q`` is
    concave, higher CTH (for fixed MTT) lowers the OEF — the heterogeneity
    penalty on oxygen delivery. Returns a fraction in ``[0, 1]``.
    """
    t, w = _flatten_finite(tof, weights)
    if t.size == 0:
        return 0.0
    return float(np.average(1.0 - np.exp(-k * t), weights=w))
