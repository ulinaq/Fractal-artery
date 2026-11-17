"""Oxygen / solute transport and extraction on the Darcy field (Krogh-style).

Steady advection–reaction(–diffusion) of a blood-borne solute (Papers B/C context;
the classic Kety/Renkin–Crone extraction picture). Blood delivers solute at the
arterial terminals with concentration ``c_art``, it is advected by the Darcy flux,
optionally diffuses through tissue, is carried out at the venous terminals, and is
consumed by tissue metabolism at a first-order rate ``k``:

    ∇·(v C) − D ∇²C + k C = delivery.

Discretised on the same finite-volume connectivity as tracer transport (shared
:func:`perfusion.transport.build_advection` / upwind operator), the steady balance
per cell is

    (A − diag(extract) − k·diag(V) − D·L) C = −inject·c_art,

with ``A`` the net advective inflow, ``inject``/``extract`` the arterial/venous
terminal weights, ``V`` the pore volume and ``L`` the TPFA diffusion Laplacian.
With ``k > 0`` the operator is a non-singular M-matrix, so ``C ≥ 0``. Along a single
streamline with no diffusion this reduces to the Kety model ``C = c_art e^{-k t(x)}``,
whose global extraction is ``1 − e^{-k·MTT}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .darcy import Grid
from .transport import build_advection


@dataclass
class OxygenSolution:
    """Steady oxygen/solute field and the derived extraction budget."""

    concentration: np.ndarray     # (nx, ny[, nz]) tissue solute concentration (NaN off-mask)
    oef: float                    # global extraction fraction = consumption / delivery
    delivery: float               # total arterial solute delivery rate
    consumption: float            # total tissue consumption rate
    c_art: float
    k: float                      # dimensionless Damköhler number (extraction / perfusion)
    rate: float = 1.0             # effective first-order rate k_eff = k·f_ref [1/time]

    @property
    def saturation(self) -> np.ndarray:
        """``C / c_art`` — fraction of the arterial level remaining (≈1 near feeds)."""
        return self.concentration / self.c_art

    @property
    def extraction_rate(self) -> np.ndarray:
        """``k_eff·C`` — local tissue solute uptake per unit volume."""
        return self.rate * self.concentration


def solve_oxygen(grid: Grid, fx: np.ndarray, fy: np.ndarray, sources,
                 terminal_flux: np.ndarray, c_art: float = 1.0, k: float = 1.0,
                 diffusivity: float = 0.0, fz: np.ndarray | None = None) -> OxygenSolution:
    """Solve the steady advection–reaction(–diffusion) solute balance.

    ``k`` is the first-order metabolic extraction rate; ``diffusivity`` scales the
    (geometric/transmissibility) TPFA diffusion Laplacian — a relative smoothing
    strength, not an absolute D. One of them must be positive for a well-posed solve.
    Pass ``fz`` (with a ``Grid3D``) for the 3-D field.
    """
    if k <= 0.0 and diffusivity <= 0.0:
        raise ValueError("need k>0 or diffusivity>0 for a well-posed steady solve")
    A, inject, extract, pore_volume = build_advection(grid, fx, fy, sources, terminal_flux, fz)
    # ``k`` is a dimensionless Damköhler number: the extraction rate relative to the
    # bulk perfusion rate f = inflow/volume = 1/MTT, so k·MTT = k and the well-mixed
    # limit gives OEF ≈ 1 − e^{−k} (matches perfusion_metrics' auto-scaled Kety rate).
    # (tissue volume = the *active* cells: an organ mask must not dilute the rate)
    active = np.asarray(grid.mask, bool).ravel(order="F")
    f_ref = float(inject.sum()) / max(float(pore_volume[active].sum()), 1e-30)
    k_eff = float(k) * f_ref
    m = A - sp.diags(extract) - k_eff * sp.diags(pore_volume)
    if diffusivity > 0.0:
        if fz is None:
            from .darcy import tpfa_operator
            lap = tpfa_operator(grid)[0]
        else:
            from .darcy3d import tpfa_operator_3d
            lap = tpfa_operator_3d(grid)
        m = m - diffusivity * lap
    conc = spla.spsolve(m.tocsr(), -inject * float(c_art))
    conc = np.clip(conc, 0.0, None)

    delivery = float(c_art) * float(inject.sum())
    consumption = k_eff * float((pore_volume * conc).sum())
    oef = consumption / delivery if delivery > 0.0 else 0.0

    field = np.where(grid.mask, conc.reshape(grid.mask.shape, order="F"), np.nan)
    return OxygenSolution(field, float(oef), float(delivery), float(consumption),
                          float(c_art), float(k), float(k_eff))


def oxygen_from_result(result, c_art: float = 1.0, k: float = 1.0,
                       diffusivity: float = 0.0) -> OxygenSolution:
    """Convenience wrapper: solve the oxygen field for a :class:`HybridResult`
    (2-D) or :class:`~perfusion.core.pipeline3d.HybridResult3D` (3-D)."""
    return solve_oxygen(result.grid, result.fx, result.fy, result.sources,
                        result.solution.terminal_flux, c_art=c_art, k=k,
                        diffusivity=diffusivity, fz=getattr(result, "fz", None))
