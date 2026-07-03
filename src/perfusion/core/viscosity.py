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
