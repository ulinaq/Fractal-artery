"""Linear-solver helpers for the Darcy TPFA systems.

A direct sparse LU (SciPy) is the default and is best for the modest 2-D grids. For
large / 3-D systems an **algebraic-multigrid** (AMG) preconditioned CG scales far
better (near-linear in the number of cells); it needs the optional ``amg`` extra
(``pyamg``) and falls back to the direct solve when unavailable.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def amg_available() -> bool:
    """True if the optional ``pyamg`` backend is importable."""
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


def solve_spd(a: sp.spmatrix, b: np.ndarray, *, method: str = "direct",
              tol: float = 1e-10) -> np.ndarray:
    """Solve ``a x = b`` for one or many right-hand sides.

    ``method='direct'`` factorises once (``splu``) and back-substitutes every column —
    ideal for 2-D. ``method='amg'`` builds a smoothed-aggregation AMG hierarchy once
    and CG-solves each column (scales to large 3-D grids); requires ``pyamg`` and
    raises ``ImportError`` if it is missing. ``a`` need not be symmetric (the Dirichlet
    pin breaks symmetry); AMG uses it as a preconditioner with CG on the near-SPD part.
    """
    b = np.asarray(b, dtype=float)
    single = b.ndim == 1
    rhs = b.reshape(-1, 1) if single else b
    if method == "amg":
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(a))
        cols = [ml.solve(rhs[:, j], tol=tol, accel="cg", maxiter=500)
                for j in range(rhs.shape[1])]
        x = np.column_stack(cols)
    elif method == "direct":
        lu = spla.splu(sp.csc_matrix(a))
        x = lu.solve(rhs)
    else:
        raise ValueError(f"unknown method {method!r} (use 'direct' or 'amg')")
    return x[:, 0] if single else x
