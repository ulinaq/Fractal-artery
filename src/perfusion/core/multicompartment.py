"""Multi-compartment Darcy capillary bed (Michler / Hyde–Smith).

Replaces the single porous bed with ``N`` stacked compartments on the same grid —
e.g. arteriole → capillary → venule — each with its own permeability ``K_c`` and
coupled to its neighbours by a linear pressure-exchange (mass-transfer) term
``β (p_c − p_{c+1})``. Blood enters the *first* compartment at the arterial
terminals and leaves the *last* at the venous terminals; it percolates down the
compartment stack through the exchange, so the inter-compartment exchange field is
the capillary perfusion.

The construction reuses the existing 1-D network solver unchanged: a generalised
:func:`terminal_response_multi` places arterial sources in compartment 0 and venous
sources in compartment ``N−1``, and returns the same terminal-to-terminal response
matrix the network solver already consumes — cross terms now travel through the
compartment exchange rather than a single bed. With ``N = 1`` it is identical to the
single-bed :func:`perfusion.darcy.terminal_response`, so the hybrid model is the
one-compartment special case.

Every function is dimension-agnostic: a 3-D :class:`~perfusion.core.darcy3d.Grid3D`
uses the 3-D TPFA operator and bump source, and fields keep the mask's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .darcy import Grid, mollifier, reconstruct_field, tpfa_operator, with_permeability


def _is_3d(grid) -> bool:
    return getattr(grid, "nz", None) is not None


def _tpfa(grid) -> sp.csr_matrix:
    """The single-compartment TPFA operator for a 2-D or 3-D grid."""
    if _is_3d(grid):
        from .darcy3d import tpfa_operator_3d

        return tpfa_operator_3d(grid)
    return tpfa_operator(grid)[0]


def _source(grid, center: int, eps: float):
    """The mollified terminal source for a 2-D or 3-D grid."""
    if _is_3d(grid):
        from .darcy3d import point_source_3d

        return point_source_3d(grid, center, eps)
    return mollifier(grid, center, eps)


def multicompartment_operator(grid: Grid, permeabilities, betas) -> sp.csc_matrix:
    """Block ``(N·n) × (N·n)`` operator for ``N`` exchange-coupled Darcy compartments.

    ``permeabilities`` is the per-compartment scalar ``K_c`` (length ``N``); ``betas``
    the exchange coefficients between adjacent compartments (length ``N−1``, each
    > 0 — a zero leaves a compartment stack uncoupled and singular). A heterogeneous
    grid keeps its pattern: compartment ``c`` is the grid rescaled to ``K_c``
    (:func:`perfusion.core.darcy.with_permeability`), so lesions apply to every
    compartment. Each
    diagonal block is the TPFA operator for that compartment plus the exchange it
    participates in (on active cells only); off-diagonal blocks are the exchange.
    """
    permeabilities, betas = np.atleast_1d(permeabilities), np.atleast_1d(betas)
    if permeabilities.size < 1 or (permeabilities <= 0).any():
        raise ValueError("every compartment needs a permeability > 0")
    if betas.size != permeabilities.size - 1:
        raise ValueError(f"{permeabilities.size} compartments need "
                         f"{permeabilities.size - 1} exchange coefficients, got {betas.size}")
    if (betas <= 0).any():
        raise ValueError("exchange coefficients must be > 0 (β = 0 decouples the "
                         "compartments and the bed becomes singular)")
    n = grid.n_cells
    active = grid.mask.ravel(order="F").astype(float)
    vol = grid.cell_volume
    ls = [_tpfa(with_permeability(grid, k)) for k in permeabilities]

    n_comp = len(permeabilities)
    diag_add = [np.zeros(n) for _ in range(n_comp)]
    blocks: list[list] = [[None] * n_comp for _ in range(n_comp)]
    for c, beta in enumerate(betas):
        b = float(beta) * vol * active            # exchange only on active cells
        diag_add[c] += b
        diag_add[c + 1] += b
        blocks[c][c + 1] = -sp.diags(b)
        blocks[c + 1][c] = -sp.diags(b)
    for c in range(n_comp):
        blocks[c][c] = ls[c] + sp.diags(diag_add[c])
    return sp.bmat(blocks, format="csc")


@dataclass
class MultiCompartmentResponse:
    response: np.ndarray          # (T, T) terminal-to-terminal, for the network solver
    field_response: np.ndarray    # (N*n, T) per-compartment field response columns
    sources: list                 # per-terminal (cells, weights)
    n_compartments: int
    n_cells: int


def terminal_response_multi(grid: Grid, art_cells: np.ndarray, ven_cells: np.ndarray,
                            permeabilities, betas, eps: float) -> MultiCompartmentResponse:
    """Multi-compartment analogue of :func:`perfusion.darcy.terminal_response`.

    Arterial terminals source/read compartment 0; venous terminals source/read the
    last compartment. Returns a response matrix (arterial terminals first, then
    venous — matching ``pipeline._terminal_cells``) plus the full ``(N·n, T)`` field
    response for reconstructing every compartment.
    """
    n_comp, n = len(permeabilities), grid.n_cells
    from .darcy import pin_row

    pin = int(np.flatnonzero(grid.mask.ravel(order="F"))[0])   # first active cell, comp 0
    lu = spla.splu(pin_row(multicompartment_operator(grid, permeabilities, betas), pin))

    all_cells = np.concatenate([np.asarray(art_cells, int), np.asarray(ven_cells, int)])
    read = np.array([0] * len(art_cells) + [n_comp - 1] * len(ven_cells))
    sources = [_source(grid, int(c), eps) for c in all_cells]

    t = len(all_cells)
    rhs = np.zeros((n_comp * n, t))
    for j, (cells, weights) in enumerate(sources):
        rhs[read[j] * n + cells, j] = weights
    rhs[pin, :] = 0.0
    field = lu.solve(rhs) if t else np.zeros((n_comp * n, 0))

    response = np.zeros((t, t))
    for i, (cells, weights) in enumerate(sources):
        response[i, :] = weights @ field[read[i] * n + cells, :]
    return MultiCompartmentResponse(response, field, sources, n_comp, n)


def compartment_fields(mcr: MultiCompartmentResponse, terminal_flux: np.ndarray,
                       balance: bool = True) -> np.ndarray:
    """Reconstruct every compartment's pressure field: ``(N, n_cells)``."""
    n = mcr.n_cells
    full = reconstruct_field(mcr.field_response, terminal_flux, balance=balance)
    return full.reshape(mcr.n_compartments, n)


def exchange_field(grid: Grid, comp_press: np.ndarray, betas) -> np.ndarray:
    """Inter-compartment perfusion ``β (p_c − p_{c+1})`` per cell, per interface.

    Returns ``(N−1, *grid.mask.shape)``: the volumetric flow (per unit volume × cell volume)
    percolating from each compartment to the next — the capillary perfusion field.
    """
    active = grid.mask.ravel(order="F")
    out = []
    for c, beta in enumerate(betas):
        e = float(beta) * grid.cell_volume * (comp_press[c] - comp_press[c + 1])
        e = np.where(active, e, np.nan)
        out.append(e.reshape(grid.mask.shape, order="F"))
    return np.array(out)
