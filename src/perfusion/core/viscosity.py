"""Blood viscosity in the microvasculature (Fahraeus-Lindqvist effect).

Implements the radius-dependent apparent viscosity of Secomb & Pries, as used in
the thesis (Eqs. 4.5-4.7). The apparent viscosity drops as vessel diameter
approaches the size of red blood cells, so small vessels contribute
disproportionately to total resistance.
"""

from __future__ import annotations

import warnings

import numpy as np

# Below this diameter the in-vivo law leaves its range: (D/(D−1.1))² is singular at
# D = 1.1 µm and the curve turns back down beneath it. No real capillary is narrower
# than ~3 µm, so the diameter is clamped there.
FL_MIN_DIAMETER_UM = 3.0


def apparent_viscosity(diameter_um: np.ndarray | float, hematocrit: float = 0.45,
                       mu_plasma: float = 1.0) -> np.ndarray:
    """Relative apparent blood viscosity as a function of vessel diameter.

    Parameters
    ----------
    diameter_um : vessel diameter in micrometres (D = 2r).
    hematocrit : discharge hematocrit H_D (0.45 reference).
    mu_plasma : plasma viscosity multiplier (relative units by default).

    Returns
    -------
    mu : apparent viscosity (same shape as ``diameter_um``), in units of
        ``mu_plasma``.

    Notes
    -----
    Thesis Eqs. (4.5)-(4.7) (Pries & Secomb in-vivo law). Diameters below
    :data:`FL_MIN_DIAMETER_UM` are clamped to it, with a warning when most vessels
    are that small (usually a wrong ``radius_to_um``): the formula is singular at
    D = 1.1 µm.
    """
    d = np.asarray(diameter_um, dtype=float)
    small = d < FL_MIN_DIAMETER_UM
    if small.size and small.mean() > 0.5:
        warnings.warn(f"Fåhræus–Lindqvist: {small.mean():.0%} of vessels are narrower than "
                      f"{FL_MIN_DIAMETER_UM:g} µm and were clamped to it — check "
                      "radius_to_um (model radius → µm)", RuntimeWarning, stacklevel=2)
    d = np.maximum(d, FL_MIN_DIAMETER_UM)
    # In-vitro reference viscosity at H_D = 0.45 (Eq. 4.6)
    mu045 = 6.0 * np.exp(-0.085 * d) + 3.2 - 2.44 * np.exp(-0.06 * d**0.645)
    # Shape parameter C (Eq. 4.7)
    poly = 1.0 + 1e-11 * d**12
    C = (0.8 + np.exp(-0.075 * d)) * (-1.0 + 1.0 / poly) + 1.0 / poly
    # Diameter correction factor (Eq. 4.5)
    corr = (d / (d - 1.1)) ** 2
    frac = ((1.0 - hematocrit) ** C - 1.0) / ((1.0 - 0.45) ** C - 1.0)
    mu_rel = (1.0 + (mu045 - 1.0) * frac * corr) * corr
    return mu_plasma * mu_rel


# --------------------------------------------------------------------------- #
# Shear-thinning blood (Carreau law)
# --------------------------------------------------------------------------- #
# Whole blood is non-Newtonian: red cells aggregate (rouleaux) at low shear, so the
# viscosity rises as flow slows and falls back to its high-shear (Newtonian) value as
# shear breaks the aggregates. The Carreau law (Cho & Kensey 1991 fit for blood:
# mu_0 = 56 mPa s, mu_inf = 3.45 mPa s, lambda = 3.313 s, n = 0.3568) captures it:
#     mu(g) = mu_inf + (mu_0 - mu_inf) (1 + (lambda g)^2)^((n - 1)/2).
# Here it multiplies the Newtonian (high-shear) viscosity, so ``mu0_ratio`` = mu_0/mu_inf.
CARREAU_N = 0.3568
CARREAU_MU0_RATIO = 56.0 / 3.45


def wall_shear_rate(q, radius) -> np.ndarray:
    """Poiseuille wall shear rate ``4 |Q| / (pi r^3)`` per segment."""
    r = np.asarray(radius, dtype=float)
    return 4.0 * np.abs(np.asarray(q, dtype=float)) / (np.pi * r**3)


def carreau_factor(shear_rate, lam: float = 1.0, n: float = CARREAU_N,
                   mu0_ratio: float = CARREAU_MU0_RATIO) -> np.ndarray:
    """Carreau viscosity relative to the high-shear value: ``1`` at high shear,
    ``mu0_ratio`` at rest, ``∝ (lam g)^(n-1)`` in between (shear-thinning for ``n < 1``).

    ``lam`` is the Carreau time constant in model time units — ``lam × shear rate`` ≈ 1
    marks where thinning sets in."""
    if lam < 0 or not 0 < n <= 1 or mu0_ratio < 1:
        raise ValueError("Carreau law needs lam >= 0, 0 < n <= 1 and mu0_ratio >= 1")
    g = np.asarray(shear_rate, dtype=float)
    return 1.0 + (mu0_ratio - 1.0) * (1.0 + (lam * g) ** 2) ** ((n - 1.0) / 2.0)


CASSON_MAX_FACTOR = 1.0e4      # cap on the plug-flow viscosity of a vessel below yield


def casson_flow_ratio(xi) -> np.ndarray:
    """Casson tube flow relative to Newtonian flow at the same wall stress:
    ``1 − (16/7)√ξ + (4/3)ξ − (1/21)ξ⁴`` with ``ξ = τ_y / τ_w`` (0 for ``ξ ≥ 1`` — the
    whole cross-section is an unyielded plug and the vessel does not flow)."""
    x = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    g = 1.0 - (16.0 / 7.0) * np.sqrt(x) + (4.0 / 3.0) * x - x**4 / 21.0
    return np.where(x < 1.0, np.maximum(g, 0.0), 0.0)


def casson_factor(shear_rate, yield_rate,
                  max_factor: float = CASSON_MAX_FACTOR) -> np.ndarray:
    """Casson apparent viscosity of a tube relative to its high-shear (Casson)
    viscosity ``μ_c``, as a function of the Poiseuille wall shear rate
    ``G = 4|Q|/(πr³)`` of the flow it carries.

    ``yield_rate`` is ``τ_y / μ_c`` (yield stress over high-shear viscosity; a scalar
    or per segment). The exact Casson tube flow at wall stress ``τ_w`` is the
    Newtonian one times :func:`casson_flow_ratio` ``(τ_y/τ_w)``; with ``t = τ_w/τ_y``
    that gives ``G / yield_rate = t · ratio(1/t) =: h(t)``, which is inverted for ``t``
    (Newton — ``h`` is increasing and convex on ``t ≥ 1``) and the factor is
    ``t / h(t)``: ``1`` with no yield stress, growing as the flow slows, capped at
    ``max_factor`` as ``Q → 0`` (the plug that does not flow below yield)."""
    yr = np.asarray(yield_rate, dtype=float)
    if np.any(yr < 0) or max_factor < 1:
        raise ValueError("Casson law needs yield_rate >= 0 and max_factor >= 1")
    g = np.abs(np.asarray(shear_rate, dtype=float))
    g, yr = np.broadcast_arrays(g, yr)
    if not np.any(yr > 0):
        return np.ones_like(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(yr > 0, g / np.where(yr > 0, yr, 1.0), np.inf)
    y_fin = np.where(np.isfinite(y), y, 0.0)          # no yield stress → factor 1 below
    t = (np.sqrt(y_fin) + 8.0 / 7.0) ** 2 + 2.0      # right of the root: Newton is monotone
    for _ in range(60):
        h = t - (16.0 / 7.0) * np.sqrt(t) + 4.0 / 3.0 - 1.0 / (21.0 * t**3)
        dh = 1.0 - (8.0 / 7.0) / np.sqrt(t) + 1.0 / (7.0 * t**4)
        step = (h - y_fin) / np.maximum(dh, 1e-300)
        t = np.maximum(t - step, 1.0)
        if np.all(np.abs(step) <= 1e-13 * t):
            break
    with np.errstate(divide="ignore"):
        f = np.where(y_fin > 0, t / np.where(y_fin > 0, y_fin, 1.0), np.inf)
    f = np.where(np.isfinite(y), f, 1.0)              # yield_rate 0: Newtonian
    return np.minimum(f, max_factor)


def solve_shear_thinning(solve, art, ven, base_viscosity, lam: float = 1.0,
                         n: float = CARREAU_N, mu0_ratio: float = CARREAU_MU0_RATIO,
                         max_iter: int = 100, tol: float = 1e-6, law: str = "carreau",
                         yield_stress: float = 0.0,
                         max_factor: float = CASSON_MAX_FACTOR, relax: float = 1.0,
                         accelerate: bool | None = None):
    """Solve the network with **non-Newtonian** blood by fixed-point iteration.

    ``solve(art, ven, (mu_art, mu_ven))`` is any hybrid / direct solve returning
    ``q_a``/``q_v``; ``base_viscosity`` is the Newtonian (high-shear) viscosity — a
    scalar or a per-tree ``(mu_art, mu_ven)`` pair (e.g. Fåhræus–Lindqvist per segment).

    ``law="carreau"`` (shear-thinning): each step sets every segment's viscosity to
    ``base × carreau_factor(wall shear rate)`` from the previous flows and re-solves
    (Picard). In the power-law range the update contracts by ``1 − n`` per step, so it
    converges for any ``n`` in ``(0, 1]``.

    ``law="casson"`` (yield stress ``yield_stress`` = τ_y, in pressure units; the base
    viscosity is the Casson high-shear viscosity): each step sets every segment's
    factor to the secant viscosity of the exact Casson tube law at its current flow
    (:func:`casson_factor`), so a converged solution carries exactly the Casson flow
    for its pressure drop. A vessel whose flow vanishes becomes a plug (factor
    ``max_factor``). This Picard step is a majorise-minimise step on the network's
    (convex) dissipation, so it converges; it is Anderson-accelerated because it
    crawls where many vessels sit near yield.

    Returns ``(solution, (mu_art, mu_ven), info)`` with ``info = {iterations, converged,
    factor_min, factor_max, law}`` (the factor range over both trees) plus, for Casson,
    ``plugged`` — the number of segments below yield.
    """
    import warnings

    if law not in ("carreau", "casson"):
        raise ValueError(f"unknown blood law {law!r} (use 'carreau' or 'casson')")
    if isinstance(base_viscosity, (tuple, list)) and len(base_viscosity) == 2:
        mu_a0, mu_v0 = base_viscosity
    else:
        mu_a0 = mu_v0 = base_viscosity
    mu_a0 = np.broadcast_to(np.asarray(mu_a0, float), (art.n_segments,))
    mu_v0 = np.broadcast_to(np.asarray(mu_v0, float), (ven.n_segments,))
    if not 0 < relax <= 1:
        raise ValueError("relax must be in (0, 1]")
    if yield_stress < 0:
        raise ValueError("yield_stress must be >= 0")

    def factors(sol):
        ra = wall_shear_rate(sol.q_a, art.seg_radius)
        rv = wall_shear_rate(sol.q_v, ven.seg_radius)
        if law == "carreau":
            return (carreau_factor(ra, lam, n, mu0_ratio), carreau_factor(rv, lam, n, mu0_ratio))
        return (casson_factor(ra, yield_stress / mu_a0, max_factor),
                casson_factor(rv, yield_stress / mu_v0, max_factor))

    na_ = art.n_segments
    sol = solve(art, ven, (mu_a0, mu_v0))                # Newtonian seed (high shear)
    x = np.log(np.concatenate(factors(sol)))             # unknowns: ln(viscosity factor)
    if accelerate is None:
        accelerate = law == "casson"
    hist_x, hist_r = [], []
    converged, iterations, change, previous = False, 0, np.inf, np.inf
    best, since_best = np.inf, 0
    while iterations < max(1, max_iter):
        iterations += 1
        f = np.exp(x)
        sol = solve(art, ven, (mu_a0 * f[:na_], mu_v0 * f[na_:]))
        r = np.log(np.concatenate(factors(sol))) - x     # fixed-point residual
        change = float(np.max(np.abs(r), initial=0.0))
        if change <= tol:                 # sol was solved with exp(x): self-consistent
            converged = True
            break
        x_next = x + relax * r
        if change < best:
            best, since_best = change, 0
        else:
            since_best += 1
        if since_best > 10:               # Anderson stalled (plugs flipping at the cap):
            accelerate = False            # finish with plain, always-convergent Picard
        if accelerate:
            # Anderson acceleration (depth 5) of the Picard map. Picard alone is a
            # majorise-minimise step on the dissipation, so it always converges, but
            # crawls where many vessels sit near yield; Anderson extrapolates over the
            # last steps; its history restarts (a plain step) whenever the residual
            # grows, and it is dropped for good if it stops making progress.
            if change > previous:
                hist_x.clear()
                hist_r.clear()
            hist_x.append(x.copy())
            hist_r.append(r.copy())
            del hist_x[:-6], hist_r[:-6]
            if len(hist_r) > 1:
                dr = np.diff(np.asarray(hist_r), axis=0).T
                dx = np.diff(np.asarray(hist_x), axis=0).T
                gamma, *_ = np.linalg.lstsq(dr, r, rcond=None)
                x_aa = x + r - (dx + dr) @ gamma
                if np.all(np.isfinite(x_aa)):
                    x_next = np.clip(x_aa, 0.0, np.log(max_factor))
        x, previous = x_next, change
    f = np.exp(x)
    fa, fv = f[:na_], f[na_:]
    if not converged:
        warnings.warn(f"{law} blood viscosity did not converge in {max_iter} "
                      f"iterations (last change {change:.2e})", RuntimeWarning, stacklevel=2)
    both = np.concatenate([fa, fv])
    info = {"law": law, "iterations": iterations, "converged": converged,
            "factor_min": float(both.min()) if both.size else 1.0,
            "factor_max": float(both.max()) if both.size else 1.0}
    if law == "casson":
        info["plugged"] = int(np.count_nonzero(both >= max_factor * (1 - 1e-9)))
    return sol, (mu_a0 * fa, mu_v0 * fv), info
