# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

A Python port of the multiscale blood-flow/perfusion models from U. N. A. Qohar's
PhD thesis (University of Bergen, 2023). The original MATLAB research code lives
in `legacy/` (reference only — do not extend it). New work goes in
`src/perfusion/`.

**Layering** (keep it strict — see `docs/ROADMAP.md`): `perfusion.core` is the
pure compute engine (NumPy/SciPy only, no I/O/viz/web deps); `io`, `viz`,
`segmentation`, `optimize` are service layers built on it; `api` (FastAPI
back-end — see `perfusion/api/`) and the planned web `frontend/` sit on top.
Physics goes in `core`; anything with
optional deps or side effects stays out of it. The public API (`from perfusion
import Grid, Parameters, run_hybrid, …` and module handles `perfusion.darcy`,
`perfusion.geometry`, …) is re-exported from `perfusion/__init__.py`; deep
imports use `perfusion.core.*`.

## Environment

- Dependency + venv management is **uv**. Do not use pip/conda directly.
  - `uv sync` / `uv sync --extra image` / `uv sync --extra api` — install.
  - `uv run pytest` — tests.  `uv run ruff check src tests scripts` — lint.
  - `uv run perfusion-api` — start the FastAPI back-end.
  - `uv run <cmd>` — run anything inside the project venv.
- Target Python ≥ 3.10. The package is importable as `perfusion` (src layout).
- **Optional extras keep the core light**: `image` (scikit-image, for
  `segmentation`), `api` (fastapi/uvicorn/pydantic, for `perfusion.api`). Import
  these lazily; never import them from `core`. Tests that need them use
  `pytest.importorskip`.

## Architecture (data flow)

```
Parameters ──► geometry.arterial_tree / venous_tree ──► Network
                                                          │
Grid ──► darcy.terminal_response (TPFA + mollifier) ──► response, field_response, sources
                                                          │
   network.solve_linear / solve_nonlinear (response, p_in, p_out) ──► HybridSolution
                                                          │
        darcy.reconstruct_field ──► pressure ──► darcy.face_fluxes
                                                          │
                       timeofflight.{flux_magnitude, time_of_flight}
```

`pipeline.run_hybrid` chains all of the above. The direct-connection model
(`direct.solve_direct`) is an alternative to the Darcy coupling that closes the
network with resistive links instead of a porous bed.

## Conventions that matter

- **Grid indexing is column-major** (`order="F"`), matching MATLAB
  `reshape`/`sub2ind`. `Grid.sub2ind(ix, iy) = ix + iy*nx`. Keep this consistent
  or fields transpose silently.
- **Network arrays are 0-based** (MATLAB was 1-based). `seg_start`/`seg_end` hold
  node indices; `incoming_segment[node]` is the feeding segment (−1 at the root);
  `node_role` uses the `NodeRole` enum (ROOT / BIFURCATION / TERMINAL).
- **Incidence-matrix assembly**: Poiseuille and mass conservation are built from
  the signed segment-node incidence `Network.incidence()` (`S`), not per-node
  loops. `(S @ p)[s] = p_start − p_end`; `(S.T @ q)[node]` is net outflow. Both
  the hybrid and direct solvers assemble the system as sparse blocks — prefer
  extending that pattern over element-wise triplet loops.
- **Vein flow orientation**: `q_v[s]` is positive toward the venous root, so the
  venous Poiseuille row is `−(S @ p_v)[s] − R q = 0` (i.e. `p_v[end] − p_v[start] = R q`).
- **Shared upwind connectivity**: `darcy.upwind_operator` returns the net-inflow
  operator + per-cell outflow; both `timeofflight` and `transport` build on it.
- **Boundary conditions**: Dirichlet `p_in` (arterial root) and `p_out` (venous
  root). The Darcy operator is singular; one cell is pinned and terminal fluxes
  are rebalanced (`reconstruct_field(balance=True)`) so the pin carries no net
  flux.
- **Thesis equation numbers** are cited in docstrings — preserve them when
  editing so the code stays traceable to the source.

## Code style & linting

Ruff enforces a broad ruleset (`E,F,I,UP,B,SIM,RUF,PERF,NPY,PT`) — run
`uv run ruff check src tests scripts` and keep it clean. Prefer modern NumPy
(`np.random.default_rng`, `np.flatnonzero`), vectorised array ops over Python
loops, `_` for unused unpacked values, and `pytest.raises(..., match=…)`.

## Verifying changes

Physics sanity checks live in `tests/` (one file per module, shared fixtures in
`tests/conftest.py`):
- single-vessel Poiseuille `ΔP = RQ` (`test_geometry.py`),
- Darcy reproduces a linear pressure ramp (`test_darcy.py`),
- hybrid + direct solvers conserve mass, incl. arterial-in == venous-out
  (`test_network.py`, `test_direct.py`, `test_pipeline.py`),
- masked Darcy isolates the organ; mollifier stays finite (`test_darcy.py`),
- TKM round-trips a known `(Fp, vp)` (`test_tkm.py`),
- a locked regression vs `tests/data/reference_hybrid.npz` (`test_pipeline.py`).

After any change to the solvers or grid conventions, run `uv run pytest` and
`uv run perfusion-tree` (should print positive trunk inflow and finite TOF).

## Common extensions (good first tasks)

- **Vessel-wall elasticity** (Eq. 4.9): make segment radii pressure-dependent
  inside `network.solve_nonlinear` (currently only junction losses are active).
- **Optimisation driver**: add a `scipy.optimize`/particle-swarm loop over
  `optimize.objective` for the Paper D flux-vs-TOF comparison.
- **Heterogeneous permeability**: `Grid.permeability` is currently a scalar;
  generalise `transmissibilities` to a per-cell field (harmonic mean).

## Recent improvements (keep these invariants)

- **Mask is honoured by the Darcy operator**: `transmissibilities` zeroes faces
  touching inactive cells and `tpfa_operator` gives inactive cells identity rows,
  so the active domain is hydraulically isolated (organ-shaped, not the bounding
  box). `time_of_flight` returns NaN on inactive cells. The pin in
  `terminal_response` is the first *active* cell.
- **Fåhræus–Lindqvist viscosity** is wired in via `Parameters.fahraeus_lindqvist`
  (+`radius_to_um`); `run_hybrid` then passes a per-tree `(mu_art, mu_ven)` pair
  of per-segment viscosities to the solvers. `_resistances` accepts a scalar or a
  per-tree pair.
- **Nonlinear solve is a fixed-point** that reuses one LU factorisation of the
  linear operator (`solve_nonlinear`), ~12× faster than the old dense-Jacobian
  Newton and reducing exactly to `solve_linear` when `junction_loss=False`.
- **Tracer transport uses the theta rule** (`transport.transport_forward(theta=)`):
  `theta=0` explicit (CFL-checked, warns), `0.5` Crank–Nicolson (default), `1`
  implicit Euler. `transport.cfl_limit` gives the explicit stability bound.
- **`direct.connect_terminals`** uses `scipy.spatial.cKDTree`; coincident terminal
  cells are spread to distinct active cells in `pipeline._spread_coincident`.
- **Image → network segmentation** (`segmentation.py`): threshold → skeletonise →
  trace graph → radius from the distance transform → BFS-rooted `Network`.
  `load_frog_system()` segments the bundled artery/vein images + tongue mask into
  a coupled system; `run_hybrid(..., art=, ven=)` accepts prebuilt networks, so
  `perfusion-frog` now reproduces Paper A on the *real* segmented vasculature.
  The tracer keeps the largest connected component and renumbers nodes so the
  solver stays non-singular; venous graphs legitimately contain loops.

## Gotchas

- The mollifier uses the thesis power-1 bump `exp(-1/(1-r²))` (not `^8`), restricts
  support to `Grid.mask`, and falls back to uniform weights on a degenerate support.
- `optimize.objective(strict=True)` (and `run_hybrid(strict_domain=True)`) returns
  `inf`/raises for trees that leave the grid; the default clips instead.
- The regression test locks to `tests/data/reference_hybrid.npz` — after any change
  that legitimately shifts the default numerics, rerun `scripts/generate_data.py`.
- Large `.mat` files in `legacy/` (whole-brain data) are only for reference; the
  package does not depend on them.

## Generating the PDF

PDFs are produced by **pandoc (Markdown → styled HTML) → headless Chrome
(HTML → PDF)** — Chrome's `Skia/PDF` engine, the same producer as the existing
files. Note: pandoc 2.x has no `--embed-resources`; inline the CSS with
`--include-in-header` instead.

```bash
# 1. Stylesheet + inline-header wrapper (once)
cat > /tmp/review.css <<'CSS'
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 3px solid #2563eb; padding-bottom: 6px; }
h2 { font-size: 15pt; margin-top: 1.4em; border-bottom: 1px solid #d1d5db; padding-bottom: 3px; }
h3 { font-size: 12.5pt; color: #374151; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #cbd5e1; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #eff6ff; }
tr:nth-child(even) td { background: #f8fafc; }
code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
blockquote { border-left: 4px solid #f59e0b; background: #fffbeb; margin: 10px 0; padding: 6px 14px; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.4em 0; }
CSS
{ printf '<style>\n'; cat /tmp/review.css; printf '\n</style>\n'; } > /tmp/review_header.html

# 2. Per review doc: Markdown -> HTML -> PDF
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for base in SPRINKLER_ENGINEERING_REVIEW VENTILATION_ENGINEERING_REVIEW; do
  pandoc "$base.md" -f gfm -s --metadata title="$base" \
    --include-in-header=/tmp/review_header.html -o "/tmp/$base.html"
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$base.pdf" "/tmp/$base.html"
done
```


Requires `pandoc` and Google Chrome. A correct run yields ~450–650 KB PDFs; a
~45 KB output means pandoc errored and Chrome printed an empty/error page.