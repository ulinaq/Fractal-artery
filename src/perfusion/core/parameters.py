"""Model parameters for the multiscale blood-flow framework.

Consolidates the scattered ``*_parameter.m`` files (Frog_parameter, Tree_parameter)
into a single dataclass with sensible defaults and a helper for the bifurcation
geometry (angles + child/parent radius ratios) from Murray's law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bifurcation_geometry(gamma: float, exponent: float) -> tuple[np.ndarray, np.ndarray]:
    """Bifurcation angles and radius ratios for a parent -> (child1, child2) split.

    Parameters
    ----------
    gamma : asymmetry ratio (r1/r2)^2, in [0, 1] with 1 = symmetric.
    exponent : Murray bifurcation exponent xi (2 turbulent .. 3 laminar).

    Returns
    -------
    phi : (2,) angles [rad] between each child and the parent axis.
    ratio : (2,) child/parent radius ratios [delta, beta].

    Notes
    -----
    Thesis Eq. (4) of Paper D and Frog_parameter.m ``phiratio``.
    """
    delta = (1.0 + gamma ** (exponent / 2.0)) ** (-1.0 / exponent)
    beta = delta * np.sqrt(gamma)
    phi1 = np.arccos((1.0 + delta**4 - beta**4) / (2.0 * delta**2))
    phi2 = np.arccos((1.0 - delta**4 + beta**4) / (2.0 * beta**2))
    return np.array([phi1, phi2]), np.array([delta, beta])


@dataclass
class Parameters:
    """Blood, flow, and vascular-geometry parameters.

    Units follow the thesis (CGS-like, non-dimensionalised where noted). The
    defaults reproduce the synthetic fractal-tree experiments of Paper D; the
    frog experiments (Paper A) override ``p_in``/``q_in`` and radii.
    """

    # Blood / flow
    rho: float = 1.0          # density
    mu: float = 1.0           # (plasma) viscosity
    p_in: float = 10.0        # arterial inlet (Dirichlet) pressure
    p_out: float = 0.0        # venous outlet (Dirichlet) pressure
    q_in: float = 10.0        # arterial inlet volumetric flow (Neumann alternative)
    hematocrit: float = 0.45

    # Radius-dependent viscosity (Fahraeus-Lindqvist). When enabled, each
    # segment's viscosity is mu * apparent_viscosity(2 r * radius_to_um, HD).
    fahraeus_lindqvist: bool = False
    radius_to_um: float = 1.0  # model-radius -> micrometre conversion for the FL law

    # Arterial geometry
    r0: float = 2.0           # initial (root) radius
    length_to_radius: float = 10.0
    exponent: float = 2.8     # Murray bifurcation exponent xi
    r_min_frac: float = 0.15  # stop branching below r_min_frac * r0 (tree depth)
    a_init: tuple[float, float] = (35.0, 0.0)
    a_theta: float = np.pi / 2
    a_gamma: float = 1.4      # root asymmetry

    # Venous geometry (relative to arterial)
    rv_scale: float = 1.2     # vein/artery radius ratio
    v_init: tuple[float, float] = (30.0, 160.0)
    v_theta: float = np.pi / 2
    v_gamma: float = 1.2

    # Reproducibility for the randomised deeper-generation asymmetries
    seed: int | None = 0

    # Derived (filled in __post_init__)
    r_min: float = field(init=False)
    rv_min: float = field(init=False)
    phi: np.ndarray = field(init=False)
    ratio: np.ndarray = field(init=False)
    vphi: np.ndarray = field(init=False)
    vratio: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.r_min = self.r_min_frac * self.r0
        self.rv_min = self.rv_scale * self.r_min
        rng = np.random.default_rng(self.seed)

        phi0, ratio0 = bifurcation_geometry(self.a_gamma, self.exponent)
        vphi0, vratio0 = bifurcation_geometry(self.v_gamma, self.exponent)
        phis, ratios = [phi0], [ratio0]
        vphis, vratios = [vphi0], [vratio0]
        # Deeper generations use randomised asymmetry (mirrors the .m loops).
        for _ in range(5):
            p, r = bifurcation_geometry(rng.random(), self.exponent)
            phis.append(p)
            ratios.append(r)
        for _ in range(5):
            p, r = bifurcation_geometry(rng.random(), self.exponent)
            vphis.append(p)
            vratios.append(r)
        self.phi = np.array(phis)
        self.ratio = np.array(ratios)
        self.vphi = np.array(vphis)
        self.vratio = np.array(vratios)

    @property
    def rv0(self) -> float:
        return self.rv_scale * self.r0
