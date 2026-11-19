"""Direct-connection microcirculation model (Paper C).

Instead of a continuum Darcy bed, unresolved vessels (arterioles, capillaries,
venules) are represented by *direct connections*: extra resistive links joining
each arterial terminal to nearby venous terminals within an effective radius.
Each link carries a relative capillary resistance ``gamma`` derived analytically
from an auxiliary branching tree (thesis Eqs. 4.23-4.34).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .geometry import Network


def relative_capillary_resistance(n_generations: int = 1) -> float:
    """Relative capillary resistance ``gamma_n`` for an n-generation auxiliary tree.

    Thesis Eq. (4.34): ``gamma_n = sum 2^((3i-1)/3) / sum 2^((1-i)/3)``.
    ``n_generations = 1`` (the thesis default) gives ``2^(2/3) ~ 1.587``.
    """
    i = np.arange(1, n_generations + 1)
    return float(np.sum(2.0 ** ((3 * i - 1) / 3.0)) / np.sum(2.0 ** ((1 - i) / 3.0)))


@dataclass
class Connection:
    """A resistive link from an arterial terminal to a venous terminal."""

    arterial_terminal: int
    venous_terminal: int
    length: float
    radius: float
    resistance: float


def connect_terminals(art: Network, ven: Network, rho_c: float,
                      viscosity: float = 1.0, n_generations: int = 1) -> list[Connection]:
    """Build direct connections between arterial and venous terminals.

    Each arterial terminal links to every venous terminal within Euclidean
    distance ``rho_c`` — measured in 3-D when both trees carry ``node_z`` — (or,
    failing that, its single nearest venous terminal).
    Connection radius follows Murray's law, ``r_k = r_terminal * Nc^(-1/3)``,
    where ``Nc`` is the number of connections from that arterial terminal.
    """
    gamma = relative_capillary_resistance(n_generations)
    ta, tv = art.terminal_nodes, ven.terminal_nodes
    if art.is_3d != ven.is_3d:
        raise ValueError("the arterial and venous trees must both be 2-D or both 3-D")
    art_xy, ven_xy = art.positions(ta), ven.positions(tv)

    # Radius search with a KD-tree: O(T log T) instead of the O(T^2) double loop.
    ven_tree = cKDTree(ven_xy)
    within = ven_tree.query_ball_point(art_xy, rho_c)

    connections: list[Connection] = []
    for i, a in enumerate(ta):
        near = within[i]
        if not near:                                   # fall back to the single nearest
            near = [int(ven_tree.query(art_xy[i])[1])]
        r_terminal = art.seg_radius[art.leaf_segments([a])[0][0]]
        radius = r_terminal * len(near) ** (-1.0 / 3.0)
        for j in near:
            length = float(np.linalg.norm(art_xy[i] - ven_xy[j]))
            resistance = gamma * 8.0 * viscosity * length / (np.pi * radius**4)
            connections.append(Connection(int(a), int(tv[j]), length,
                                           float(radius), float(resistance)))
    return connections


@dataclass
class DirectSolution:
    p_a: np.ndarray
    q_a: np.ndarray
    p_v: np.ndarray
    q_v: np.ndarray
    q_c: np.ndarray                 # connection flows
    connections: list[Connection]


def solve_direct(art: Network, ven: Network, connections: list[Connection],
                 p_in: float, p_out: float,
                 viscosity: float | np.ndarray = 1.0) -> DirectSolution:
    """Solve the vascular network closed by direct connections.

    Unknowns ``[p_a, q_a, p_v, q_v, q_c]``. Uses the same incidence-matrix
    assembly as the hybrid solver, with the connection links replacing the Darcy
    coupling. Mass is conserved at every terminal.
    """
    Ra, Rv = art.resistance(viscosity), ven.resistance(viscosity)
    Sa, Sv = art.incidence(), ven.incidence()
    ta, tv = art.terminal_nodes, ven.terminal_nodes
    na, ma = art.n_nodes, art.n_segments
    nv, mv = ven.n_nodes, ven.n_segments
    nc = len(connections)

    pa, qa, pv, qv, qc = 0, na, na + ma, na + ma + nv, na + ma + nv + mv
    size = qc + nc

    def pad(block, col0, width, n_rows=None):
        n_rows = block.shape[0] if n_rows is None else n_rows
        left = sp.csr_matrix((n_rows, col0))
        right = sp.csr_matrix((n_rows, size - col0 - width))
        return sp.hstack([left, block, right], format="csr")

    # connection-terminal incidence: which connections touch each terminal
    a_idx = {int(t): k for k, t in enumerate(ta)}
    v_idx = {int(t): k for k, t in enumerate(tv)}
    ca_rows = [a_idx[c.arterial_terminal] for c in connections]
    cv_rows = [v_idx[c.venous_terminal] for c in connections]
    conn_col = np.arange(nc)
    Ca = sp.csr_matrix((np.ones(nc), (ca_rows, conn_col)), shape=(len(ta), nc))
    Cv = sp.csr_matrix((np.ones(nc), (cv_rows, conn_col)), shape=(len(tv), nc))

    # flow reaching each terminal through its (single) segment: orient · q, where
    # orient = +1 if the leaf is the segment's end node, −1 if its start node
    seg_a, orient_a = art.leaf_segments(ta)
    seg_v, orient_v = ven.leaf_segments(tv)
    sel_a = sp.csr_matrix((orient_a, (np.arange(len(ta)), seg_a)), shape=(len(ta), ma))
    sel_v = sp.csr_matrix((orient_v, (np.arange(len(tv)), seg_v)), shape=(len(tv), mv))

    # one Dirichlet row per prescribed-pressure node (inlets p_in, outlets p_out)
    ia, iv = art.inlet_nodes, ven.inlet_nodes
    root_a = sp.csr_matrix((np.ones(len(ia)), (np.arange(len(ia)), pa + ia)),
                           shape=(len(ia), size))
    pois_a = pad(Sa, pa, na) + pad(-sp.diags(Ra), qa, ma)
    cons_a = pad(Sa.T.tocsr()[art.bifurcation_nodes, :], qa, ma)
    term_a = pad(sel_a, qa, ma, n_rows=len(ta)) + pad(-Ca, qc, nc, n_rows=len(ta))

    root_v = sp.csr_matrix((np.ones(len(iv)), (np.arange(len(iv)), pv + iv)),
                           shape=(len(iv), size))
    pois_v = pad(-Sv, pv, nv) + pad(-sp.diags(Rv), qv, mv)
    cons_v = pad(Sv.T.tocsr()[ven.bifurcation_nodes, :], qv, mv)
    term_v = pad(sel_v, qv, mv, n_rows=len(tv)) + pad(-Cv, qc, nc, n_rows=len(tv))

    # connection Poiseuille: p_a[a] - p_v[v] - R_k q_c[k] = 0
    a_nodes = np.array([c.arterial_terminal for c in connections], dtype=int)
    v_nodes = np.array([c.venous_terminal for c in connections], dtype=int)
    Rc = np.array([c.resistance for c in connections])
    rows = np.concatenate([conn_col, conn_col, conn_col])
    cols = np.concatenate([pa + a_nodes, pv + v_nodes, qc + conn_col])
    vals = np.concatenate([np.ones(nc), -np.ones(nc), -Rc])
    conn_pois = sp.csr_matrix((vals, (rows, cols)), shape=(nc, size))

    A = sp.vstack([root_a, pois_a, cons_a, term_a,
                   root_v, pois_v, cons_v, term_v, conn_pois], format="csr")
    b = np.zeros(size)
    b[:len(ia)] = p_in
    v0 = len(ia) + ma + len(art.bifurcation_nodes) + len(ta)  # first root_v row
    b[v0:v0 + len(iv)] = p_out

    x = spla.spsolve(A, b)
    return DirectSolution(
        p_a=x[pa:qa], q_a=x[qa:pv], p_v=x[pv:qv], q_v=x[qv:qc],
        q_c=x[qc:size], connections=connections,
    )
