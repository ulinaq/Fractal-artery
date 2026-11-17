"""Hybrid vascular-network + Darcy solver (linear and nonlinear).

Couples two vascular trees (arterial, venous) to the Darcy capillary bed through
the mollified terminal-response matrix (thesis Eq. 5.5 / 2.20). The linear model
uses Poiseuille flow + mass conservation; the nonlinear model adds the Bernoulli
bifurcation pressure loss (Eq. 4.11).

The assembly is fully vectorised: Poiseuille and mass conservation are expressed
through the signed segment-node incidence matrix ``S`` (see
:meth:`perfusion.geometry.Network.incidence`), so there are no per-node Python
loops.

Unknown layout (monolithic vector ``x``)::

    [ p_a  (na) | q_a  (ma) | p_v  (nv) | q_v  (mv) | p_ref (1) ]

``p_ref`` is a floating Darcy datum; together with the global-balance row
``sum(terminal_flux) = 0`` it closes the loop so arterial inflow equals venous
outflow (the pinned Darcy response otherwise leaks net flux through the pin).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Network


def _resistances(art: Network, ven: Network, viscosity):
    """Per-segment resistances, accepting a shared scalar or a per-tree pair.

    ``viscosity`` is either a scalar/array applied to both trees, or a
    ``(mu_art, mu_ven)`` pair (each scalar or per-segment array) -- the latter
    lets the Fahraeus-Lindqvist law give every vessel its own viscosity.
    """
    if isinstance(viscosity, (tuple, list)) and len(viscosity) == 2:
        mu_a, mu_v = viscosity
    else:
        mu_a = mu_v = viscosity
    return art.resistance(mu_a), ven.resistance(mu_v)


@dataclass
class HybridSolution:
    """Solution of the coupled network + Darcy flow problem."""

    p_a: np.ndarray            # arterial node pressures
    q_a: np.ndarray            # arterial segment flows
    p_v: np.ndarray            # venous node pressures
    q_v: np.ndarray            # venous segment flows
    terminal_flux: np.ndarray  # signed flux per terminal (+ arterial in, - venous out)
    darcy_pressure: np.ndarray | None = None   # (n_cells,) bed pressure (monolithic solve)


@dataclass
class MonolithicDarcy:
    """Sparse Darcy coupling for the **monolithic** hybrid solve.

    Instead of the dense ``(T, T)`` terminal response (and its ``(n_cells, T)`` field
    columns) the Darcy bed enters the network system directly: its cell pressures
    ``P`` become unknowns, with rows ``A P = W_src (s∘q)`` (pinned TPFA operator,
    mollified terminal sources) and coupling rows ``p_node − p_ref − W_avgᵀ P = 0``.
    Eliminating ``P`` gives back exactly the Schur form ``R = W_avgᵀ A⁻¹ W_src``, so the
    two are the same model — but memory is ``O(nnz)`` instead of ``O(n_cells · T)``,
    which is what fine (3-D) grids with thousands of terminals need.
    """

    A: sp.csr_matrix          # (n, n) TPFA operator with the Dirichlet pin row
    W_src: sp.csr_matrix      # (n, T) terminal source weights, pin row zeroed
    W_avg: sp.csr_matrix      # (n, T) terminal averaging weights (mollifier)

    @property
    def n(self) -> int:
        return self.A.shape[0]


@dataclass
class _Layout:
    """Offsets of each block in the monolithic unknown vector."""

    art: Network
    ven: Network

    def __post_init__(self):
        self.na, self.ma = self.art.n_nodes, self.art.n_segments
        self.nv, self.mv = self.ven.n_nodes, self.ven.n_segments
        self.pa = 0
        self.qa = self.na
        self.pv = self.na + self.ma
        self.qv = self.na + self.ma + self.nv
        self.p_ref = self.na + self.ma + self.nv + self.mv
        self.size = self.p_ref + 1

    def terminal_flux_columns(self):
        """Columns and signs of the terminal fluxes in ``x``.

        Ordering is [arterial terminals, venous terminals], matching the rows and
        columns of the Darcy response matrix.
        """
        ta, tv = self.art.terminal_nodes, self.ven.terminal_nodes
        art_cols = self.qa + self.art.incoming_segment[ta]
        ven_cols = self.qv + self.ven.incoming_segment[tv]
        columns = np.concatenate([art_cols, ven_cols])
        signs = np.concatenate([np.ones(len(ta)), -np.ones(len(tv))])
        return columns, signs, ta, tv


def _coupling_and_balance(lay: _Layout, response: np.ndarray,
                          columns: np.ndarray, signs: np.ndarray,
                          ta: np.ndarray, tv: np.ndarray):
    """Sparse terminal-coupling rows plus the global mass-balance row.

    Coupling row ``k``: ``p[node_k] - p_ref - sum_j response[k, j] * flux_j = 0``.
    Balance row: ``sum_j flux_j = 0``.
    """
    n_term = len(columns)
    p_cols = np.concatenate([lay.pa + ta, lay.pv + tv])  # pressure unknown per terminal

    # dense flux contribution -response[k, j] * sign_j scattered into flux columns
    krow = np.repeat(np.arange(n_term), n_term)
    kcol = np.tile(columns, n_term)
    kval = -(response * signs[None, :]).ravel()

    rows = np.concatenate([np.arange(n_term), np.arange(n_term), krow])
    cols = np.concatenate([p_cols, np.full(n_term, lay.p_ref), kcol])
    vals = np.concatenate([np.ones(n_term), -np.ones(n_term), kval])
    coupling = sp.csr_matrix((vals, (rows, cols)), shape=(n_term, lay.size))

    balance = sp.csr_matrix((signs, (np.zeros(n_term, int), columns)),
                            shape=(1, lay.size))
    return coupling, balance


def _network_blocks(net: Network, R: np.ndarray, lay: _Layout,
                    is_arterial: bool):
    """Boundary-pressure, Poiseuille and conservation rows for one tree.

    Returns ``(dirichlet, poiseuille, conservation)`` sparse blocks of width
    ``lay.size``. ``dirichlet`` has one row per inlet/outlet node (``p = p_in`` /
    ``p_out``), so any number of prescribed-pressure boundary nodes is supported.
    The venous Poiseuille uses ``-S`` (flow positive toward the root);
    conservation is ``S.T`` restricted to bifurcation nodes for both.
    """
    S = net.incidence()
    m, n = net.n_segments, net.n_nodes
    p_off = lay.pa if is_arterial else lay.pv
    q_off = lay.qa if is_arterial else lay.qv

    def pad(block, col_off, width):
        return sp.hstack([sp.csr_matrix((block.shape[0], col_off)), block,
                          sp.csr_matrix((block.shape[0], lay.size - col_off - width))],
                         format="csr")

    inlets = net.inlet_nodes
    k = len(inlets)
    dirichlet = sp.csr_matrix((np.ones(k), (np.arange(k), p_off + inlets)),
                              shape=(k, lay.size))

    sign = 1.0 if is_arterial else -1.0
    pois = pad(sign * S, p_off, n) + pad(-sp.diags(R), q_off, m)

    cons_full = S.T.tocsr()[net.bifurcation_nodes, :]
    cons = pad(cons_full, q_off, m)
    return dirichlet, pois, cons


def _coupling_monolithic(lay: _Layout, darcy: MonolithicDarcy, columns: np.ndarray,
                         signs: np.ndarray, ta: np.ndarray, tv: np.ndarray):
    """Coupling, balance and Darcy rows of the monolithic system (width
    ``lay.size + n``): ``p_node − p_ref − W_avgᵀ P = 0``, ``Σ s q = 0`` and
    ``A P − W_src (s∘q) = 0``."""
    n_term, n = len(columns), darcy.n
    width = lay.size + n
    p_cols = np.concatenate([lay.pa + ta, lay.pv + tv])
    net = sp.csr_matrix((np.concatenate([np.ones(n_term), -np.ones(n_term)]),
                         (np.concatenate([np.arange(n_term)] * 2),
                          np.concatenate([p_cols, np.full(n_term, lay.p_ref)]))),
                        shape=(n_term, lay.size))
    coupling = sp.hstack([net, -darcy.W_avg.T.tocsr()], format="csr")
    balance = sp.csr_matrix((signs, (np.zeros(n_term, int), columns)), shape=(1, width))
    # scatter terminal j's (signed) source column onto its flux unknown columns[j]
    place = sp.csr_matrix((signs, (np.arange(n_term), columns)), shape=(n_term, lay.size))
    darcy_rows = sp.hstack([-(darcy.W_src @ place), darcy.A], format="csr")
    return coupling, balance, darcy_rows


@dataclass
class _System:
    """Assembled linear hybrid system and the row slices the solvers need."""

    A: sp.csr_matrix
    b: np.ndarray
    lay: _Layout
    columns: np.ndarray
    signs: np.ndarray
    pois_a_rows: slice        # arterial Poiseuille rows (for junction-loss RHS)
    pois_v_rows: slice        # venous Poiseuille rows
    n_darcy: int = 0          # Darcy unknowns appended (monolithic solve)


def _assemble(art: Network, ven: Network, response, p_in: float,
              p_out: float, Ra: np.ndarray, Rv: np.ndarray) -> _System:
    """Assemble the sparse linear hybrid system ``A x = b``.

    ``response`` is the dense ``(T, T)`` Darcy response (Schur form) or a
    :class:`MonolithicDarcy` (the bed's cell pressures appended as unknowns)."""
    lay = _Layout(art, ven)
    columns, signs, ta, tv = lay.terminal_flux_columns()

    dir_a, pois_a, cons_a = _network_blocks(art, Ra, lay, is_arterial=True)
    dir_v, pois_v, cons_v = _network_blocks(ven, Rv, lay, is_arterial=False)
    network = [dir_a, pois_a, cons_a, dir_v, pois_v, cons_v]
    if isinstance(response, MonolithicDarcy):
        n_darcy = response.n
        coupling, balance, darcy_rows = _coupling_monolithic(lay, response, columns, signs,
                                                             ta, tv)
        network = [sp.hstack([blk, sp.csr_matrix((blk.shape[0], n_darcy))], format="csr")
                   for blk in network]
        A = sp.vstack([*network, coupling, balance, darcy_rows], format="csr")
    else:
        n_darcy = 0
        coupling, balance = _coupling_and_balance(lay, response, columns, signs, ta, tv)
        A = sp.vstack([*network, coupling, balance], format="csr")
    b = np.zeros(lay.size + n_darcy)
    ka, kv = art.inlet_nodes.size, ven.inlet_nodes.size
    nbif_a = len(art.bifurcation_nodes)
    b[:ka] = p_in                                 # dir_a: one row per arterial inlet
    v_dir = ka + lay.ma + nbif_a                  # start of the venous Dirichlet block
    b[v_dir:v_dir + kv] = p_out                   # dir_v: one row per venous outlet

    pois_a_rows = slice(ka, ka + lay.ma)
    v0 = v_dir + kv
    pois_v_rows = slice(v0, v0 + lay.mv)
    return _System(A, b, lay, columns, signs, pois_a_rows, pois_v_rows, n_darcy)


def solve_linear(art: Network, ven: Network, response: np.ndarray, p_in: float,
                 p_out: float, viscosity=1.0) -> HybridSolution:
    """Solve the linear hybrid system (Poiseuille + conservation + coupling).

    Parameters
    ----------
    art, ven : arterial and venous trees.
    response : ``(T, T)`` Darcy terminal-response matrix from
        :func:`perfusion.darcy.terminal_response`, or a :class:`MonolithicDarcy`
        (``darcy.coupling_operator``) to solve network + bed in one sparse system
        (the bed pressure is then returned in ``darcy_pressure``).
    p_in, p_out : Dirichlet pressures at the arterial/venous roots.
    viscosity : scalar/array for both trees, or a ``(mu_art, mu_ven)`` pair.
    """
    Ra, Rv = _resistances(art, ven, viscosity)
    sys = _assemble(art, ven, response, p_in, p_out, Ra, Rv)
    x = spla.spsolve(sys.A.tocsc(), sys.b)
    return _split(x, sys)


def _split(x: np.ndarray, sys: _System) -> HybridSolution:
    lay = sys.lay
    return HybridSolution(
        p_a=x[lay.pa:lay.qa], q_a=x[lay.qa:lay.pv],
        p_v=x[lay.pv:lay.qv], q_v=x[lay.qv:lay.p_ref],
        terminal_flux=sys.signs * x[sys.columns],
        darcy_pressure=x[lay.size:] if sys.n_darcy else None,
    )


def solve_nonlinear(art: Network, ven: Network, response: np.ndarray, p_in: float,
                    p_out: float, prm, viscosity=1.0, junction_loss: bool = True,
                    max_iter: int = 100, tol: float = 1e-10) -> HybridSolution:
    """Solve the nonlinear hybrid system with Bernoulli bifurcation losses.

    The only nonlinearity -- the junction pressure drop (Eq. 4.11) -- is a mild,
    flow-dependent correction to the Poiseuille balances, so we solve it as a
    fixed-point iteration that moves the correction to the right-hand side and
    **reuses a single LU factorisation** of the linear operator. This is far
    cheaper than a dense finite-difference Newton (the linear matrix never
    changes), yet reduces exactly to :func:`solve_linear` when ``junction_loss``
    is off.
    """
    Ra, Rv = _resistances(art, ven, viscosity)
    sys = _assemble(art, ven, response, p_in, p_out, Ra, Rv)
    lu = spla.splu(sys.A.tocsc())
    x = lu.solve(sys.b)                                   # linear seed

    if not junction_loss:
        return _split(x, sys)

    jl_a, jl_v = _junction_geometry(art), _junction_geometry(ven)
    lay = sys.lay
    for _ in range(max_iter):
        qa, qv = x[lay.qa:lay.pv], x[lay.qv:lay.p_ref]
        b = sys.b.copy()
        b[sys.pois_a_rows] = _junction_drop(jl_a, qa, prm.rho)     # Sa p - Ra q = dpa
        b[sys.pois_v_rows] = -_junction_drop(jl_v, qv, prm.rho)    # -(Sv p) - Rv q = -dpv
        x_new = lu.solve(b)
        if np.linalg.norm(x_new - x) <= tol * (np.linalg.norm(x_new) + tol):
            x = x_new
            break
        x = x_new
    else:
        warnings.warn("solve_nonlinear: fixed-point did not converge in "
                      f"{max_iter} iterations", RuntimeWarning, stacklevel=2)
    return _split(x, sys)


def _junction_geometry(net: Network):
    """Per-segment parent index and cos(3 dtheta/4) for the Bernoulli loss."""
    parent = net.incoming_segment[net.seg_start]     # segment feeding this segment's start
    has_parent = parent >= 0
    cos_phi = np.zeros(net.n_segments)
    p = parent[has_parent]
    cos_phi[has_parent] = np.cos(
        3.0 * (net.seg_angle[has_parent] - net.seg_angle[p]) / 4.0)
    return {"parent": parent, "has_parent": has_parent, "cos_phi": cos_phi,
            "r": net.seg_radius}


def _junction_drop(jl, q: np.ndarray, rho: float) -> np.ndarray:
    """Vectorised Bernoulli junction loss per segment (Eq. 4.11); 0 if disabled."""
    if jl is None:
        return 0.0
    dp = np.zeros_like(q)
    m = jl["has_parent"]
    parent = jl["parent"][m]
    q_p = q[parent]
    safe = np.abs(q_p) > 1e-12
    idx = np.flatnonzero(m)[safe]
    parent = parent[safe]
    q_c = q[idx]
    q_p = q_p[safe]
    r_dat = jl["r"][parent]
    r_n = jl["r"][idx]
    cphi = jl["cos_phi"][idx]
    dp[idx] = (rho * q_p**2 / (2 * np.pi**2 * r_dat**4)) * (
        1.0 + q_c**2 * r_dat**4 / (q_p**2 * r_n**4)
        - 2.0 * r_dat**2 * cphi * q_c / (q_p * r_n**2))
    return dp
