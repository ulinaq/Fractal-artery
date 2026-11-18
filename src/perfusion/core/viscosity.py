"""Blood viscosity in the microvasculature (Fahraeus-Lindqvist effect).

Implements the radius-dependent apparent viscosity of Secomb & Pries, as used in
the thesis (Eqs. 4.5-4.7). The apparent viscosity drops as vessel diameter
approaches the size of red blood cells, so small vessels contribute
disproportionately to total resistance.
"""

from __future__ import annotations

import numpy as np


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
    Thesis Eqs. (4.5)-(4.7). Valid roughly for 10 um < D < 300 um; outside that
    range it saturates smoothly.
    """
    d = np.asarray(diameter_um, dtype=float)
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


def solve_shear_thinning(solve, art, ven, base_viscosity, lam: float = 1.0,
                         n: float = CARREAU_N, mu0_ratio: float = CARREAU_MU0_RATIO,
                         max_iter: int = 100, tol: float = 1e-6):
    """Solve the network with **shear-thinning** (Carreau) blood by fixed-point iteration.

    ``solve(art, ven, (mu_art, mu_ven))`` is any hybrid / direct solve returning
    ``q_a``/``q_v``; ``base_viscosity`` is the Newtonian (high-shear) viscosity — a
    scalar or a per-tree ``(mu_art, mu_ven)`` pair (e.g. Fåhræus–Lindqvist per segment).
    Each step sets every segment's viscosity to ``base × carreau_factor(wall shear
    rate)`` from the previous flows and re-solves (Picard). In the power-law range the
    update contracts by ``1 − n`` per step, so it converges for any ``n`` in ``(0, 1]``.

    Returns ``(solution, (mu_art, mu_ven), info)`` with ``info = {iterations, converged,
    factor_min, factor_max}`` (the Carreau factor range over both trees).
    """
    import warnings

    if isinstance(base_viscosity, (tuple, list)) and len(base_viscosity) == 2:
        mu_a0, mu_v0 = base_viscosity
    else:
        mu_a0 = mu_v0 = base_viscosity
    mu_a0 = np.broadcast_to(np.asarray(mu_a0, float), (art.n_segments,))
    mu_v0 = np.broadcast_to(np.asarray(mu_v0, float), (ven.n_segments,))

    def factors(sol):
        return (carreau_factor(wall_shear_rate(sol.q_a, art.seg_radius), lam, n, mu0_ratio),
                carreau_factor(wall_shear_rate(sol.q_v, ven.seg_radius), lam, n, mu0_ratio))

    sol = solve(art, ven, (mu_a0, mu_v0))                # Newtonian seed (high shear)
    fa, fv = factors(sol)
    converged, iterations = False, 0
    while iterations < max(1, max_iter):
        iterations += 1
        sol = solve(art, ven, (mu_a0 * fa, mu_v0 * fv))
        na, nv = factors(sol)
        change = max(np.max(np.abs(np.log(na / fa)), initial=0.0),
                     np.max(np.abs(np.log(nv / fv)), initial=0.0))
        if change <= tol:                 # sol was solved with (fa, fv): self-consistent
            converged = True
            break
        fa, fv = na, nv
    if not converged:
        warnings.warn(f"shear-thinning viscosity did not converge in {max_iter} "
                      f"iterations (last change {change:.2e})", RuntimeWarning, stacklevel=2)
    both = np.concatenate([fa, fv])
    info = {"iterations": iterations, "converged": converged,
            "factor_min": float(both.min()) if both.size else 1.0,
            "factor_max": float(both.max()) if both.size else 1.0}
    return sol, (mu_a0 * fa, mu_v0 * fv), info
