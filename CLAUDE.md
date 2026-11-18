# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

A Python port of the multiscale blood-flow/perfusion models from U. N. A. Qohar's
PhD thesis (University of Bergen, 2023). The original MATLAB research code lives
in `legacy/` (reference only — do not extend it). New work goes in
`src/perfusion/`.

**Layering** (keep it strict — see `docs/ROADMAP.md`): `perfusion.core` is the
pure compute engine (NumPy/SciPy only, no I/O/viz/web deps); `io`, `viz`,
`segmentation`, `growth`, `optimize` are service layers built on it; `api`
(FastAPI back-end — see `perfusion/api/`) sits on top and also serves the
zero-build web `frontend/` — 2-D workspace, Tree explorer, 3-D viewer (plain HTML/CSS/ES-modules, mounted at `/` by
`api.app._mount_frontend`, no build step). `api.app.create_app` is thin — it sets
`app.state.store`/`jobs`, exposes `/health` + `/frontend-version`, mounts the
frontend, and includes three `APIRouter` modules grouped by concern:
`routes_projects` (lifecycle + frog + save-file document), `routes_geometry`
(vessels/mask/io/editable/vectorize/image/grow/network), `routes_solve`
(run/tracer/jobs/results/field/export/direct/view3d/demo3d). Handlers reach the
store/jobs/project through `api.deps` (`get_store`/`get_jobs`/`get_project`
+ `project_info`); shared compute lives in `api.helpers` (frog & demo 3-D scenes,
`rebuild_network`, `render_gif`). The API's `ProjectStore` persists to
SQLite + a filesystem blob store (grown/segmented networks + result & tracer
`.npz`) under `$PERFUSION_DATA`, so projects/results survive a restart; heavy
binary artefacts never go in the DB. Jobs are also in SQLite (a durable
registry): job execution lives in `api/runner.py` (+ `jobs.run_job`, which
reopens the store from its root) so it runs in either the default thread pool
(`JobManager`) or an optional RQ/Redis worker (`RQJobManager`, `PERFUSION_QUEUE=rq`),
both writing lifecycle through the store; stale in-flight jobs are reaped to
`interrupted` on startup. Physics goes in `core`; anything with optional deps or
side effects stays out of it. The public API (`from perfusion
import Grid, Parameters, run_hybrid, …` and module handles `perfusion.darcy`,
`perfusion.geometry`, …) is re-exported from `perfusion/__init__.py`; deep
imports use `perfusion.core.*`.

## Environment

- Dependency + venv management is **uv**. Do not use pip/conda directly.
  - `uv sync` / `uv sync --extra image` / `uv sync --extra api` — install.
  - `uv run pytest` — tests (api/image/mcp tests `importorskip` and are skipped
    unless their extras are synced; for the full suite:
    `uv run --extra api --extra image --extra mcp --extra video pytest`). `uv run ruff check src tests scripts` — lint.
  - `uv run perfusion-api` — start the FastAPI back-end.
  - `uv run <cmd>` — run anything inside the project venv.
- Target Python ≥ 3.10. The package is importable as `perfusion` (src layout).
- **Optional extras keep the core light**: `image` (scikit-image, for
  `segmentation`), `api` (fastapi/uvicorn/pydantic, for `perfusion.api`), `mcp`
  (the MCP SDK, for `perfusion.mcp_server`), `video` (imageio-ffmpeg, for the
  server-side `tracer.mp4` export). Import these lazily; never import
  them from `core`. Tests that need them use `pytest.importorskip`.
- **MCP server**: `perfusion.mcp_server` exposes the engine as MCP tools
  (`run_simulation`, `describe_model`) — run with `uv run --extra mcp perfusion-mcp`;
  the repo's `.mcp.json` registers it for Claude Code. Tool logic lives in plain
  functions (`simulate`, `model_info`); the `@server.tool()` wrappers are thin.
- **Expose new capabilities as MCP tools.** When you add a user-facing feature
  (a new solve/analysis mode, growth method, exporter, etc.), also surface it as
  an MCP tool in `mcp_server.py` — keep the logic in a plain, testable function
  and add a thin `@server.tool()` wrapper — so any AI client can use it too. Treat
  "is it in MCP?" as part of done for new features.
- **Model & formula reference**: `docs/MODELS.md` — every module's governing equation
  (Poiseuille/Darcy/TPFA, hybrid coupling, tracer kinetics, deconvolution, oxygen,
  adaptation, morphometry, Windkessel, UQ/Sobol) with thesis equation numbers.
- **New-contributor / extension guide**: `docs/DEVELOPING.md` (feature → files →
  pattern → tests map).

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
- **Node classification is by degree, not role.** `terminal_nodes` = degree-1
  leaves minus `inlet_nodes`; `bifurcation_nodes` = degree≥2 interior minus
  `inlet_nodes`. `Network.inlet_nodes` (default `[root_node]`) is the set of
  prescribed-pressure boundary nodes — so any leaf can be an inlet/outlet and the
  BFS root, if *not* marked, correctly becomes a Darcy terminal. Don't reintroduce
  role-based (`role==TERMINAL`) classification — it strands the root.
- **Incidence-matrix assembly**: Poiseuille and mass conservation are built from
  the signed segment-node incidence `Network.incidence()` (`S`), not per-node
  loops. `(S @ p)[s] = p_start − p_end`; `(S.T @ q)[node]` is net outflow. Both
  the hybrid and direct solvers assemble the system as sparse blocks — prefer
  extending that pattern over element-wise triplet loops.
- **Vein flow orientation**: `q_v[s]` is positive toward the venous root, so the
  venous Poiseuille row is `−(S @ p_v)[s] − R q = 0` (i.e. `p_v[end] − p_v[start] = R q`).
- **The network solver is coordinate-free** — `network.solve_linear` uses only
  topology (`incidence`), `seg_length`/`seg_radius` (via `resistance`), and
  `inlet_nodes`; it never reads `node_x/y/z` or `seg_angle`. That's why 3-D works
  by reuse: a 3-D `Network` (with `node_z`, from `geometry.grow_tree_3d`) plugs
  into the same solver, and 3-D adds only the Darcy coupling — `core.darcy3d`
  (`Grid3D`, `tpfa_operator_3d`, `point_source_3d`, `terminal_response_3d`) and
  `core.pipeline3d.run_hybrid_3d`, mirroring the 2-D `darcy`/`pipeline`.
  `reconstruct_field` is dimension-agnostic and reused as-is. Don't make the
  solver depend on coordinates.
- **Shared upwind connectivity**: `darcy.upwind_operator` returns the net-inflow
  operator + per-cell outflow; both `timeofflight` and `transport` build on it.
- **Boundary conditions**: Dirichlet `p_in` at the arterial `inlet_nodes` and
  `p_out` at the venous outlet nodes (one row per node, so multiple inlets/outlets
  are supported; `run_hybrid(art_inlets=, ven_outlets=)`). The Darcy operator is
  singular; one cell is pinned and terminal fluxes are rebalanced
  (`reconstruct_field(balance=True)`) so the pin carries no net flux.
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
  `terminal_response` is the first *active* cell. **Terminals must couple to
  active cells**: `pipeline._terminal_cells` snaps any terminal that lands on an
  inactive cell to the nearest active one (`_snap_to_active`), else its mollifier
  is degenerate and the coupling goes singular. Masks must also keep the active
  region *connected* — a mask that splits it into disconnected blobs leaves all
  but one blob as a singular Neumann block (only one cell is pinned). This is why
  the frog template uses a faithful raster resample (`segmentation.mask_in_domain`),
  not a lossy polygon outline.
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
  `network_in_domain(net, image_shape, domain_limit)` places a pixel-space
  segmented network into a physical box (column→x, row→y flipped so it reads
  upright); the API's `POST /projects/{id}/image` uses it to turn an upload into
  the arterial tree.
- **Portable save/load** (`schemas.ProjectDocument`): a whole project — params,
  grid, hand-drawn vessels, tissue regions + raster mask, I/O points and the
  segmented/grown `artery/vein` networks (with `inlet_nodes`/`node_z`, unlike the
  `.npz` blob) — round-trips losslessly as one JSON file. `GET /projects/{id}/document`
  downloads it; `POST /projects/import` restores it as a *new* project. Conversion
  lives in `store.document_from_project` / `project_from_document`; the frontend
  Project card has 💾 Save file / 📂 Load file.
- **Interactive vessel editing (network/tree model)**: vessels are edited as a
  **network** — nodes + edges, each edge carrying its own radius — *not* free
  polylines. The frontend **Vessels** tab (`frontend/edit.js`) *is* the editor:
  click empty canvas to start a vessel and keep clicking to extend it, click a
  node then empty space to branch, drag to move, click a vessel to *select the
  segment* (its radius is then editable any time), double-click a vessel to split
  it, Shift+click to join, Del to delete the selected node/segment. **Pruning**
  (`core.geometry.prune_terminal_branches`, `POST /projects/{id}/prune`, MCP
  `prune_vessel_tree`) is the inverse of growth: each step removes every terminal
  branch (free end → nearest junction through degree-2 nodes), never the chain to
  an inlet. **Inlets/outlets are network nodes**
  (`inlet_nodes`; artery → `p_in`, vein → `p_out`), toggled on the selected node —
  there is no separate I/O or Edit tab, and the old `project.inlets/outlets` points
  are only a legacy path (snapped onto nodes on open, cleared on save). `PUT
  /projects/{id}/editable` with topology rebuilds a tree via `helpers.rebuild_network`
  and can also **create** a tree the project lacks or **remove** one (zero
  segments); a move/resize PUT omits `seg_start`/`seg_end`. `_rebuild_network`
  reindexes, drops isolated non-inlet nodes and re-derives `node_role`/`seg_length`/
  `seg_angle`. `io.save_network` persists `inlet_nodes`/`node_z`, so multiple
  inlets/outlets survive a reload. `runner.validate_run` rejects a tree whose every
  end is an inlet/outlet (no terminals → no tissue exchange). Legacy hand-drawn
  polylines are folded in once by `POST /vectorize` when a project opens.
  The **Tissue** tab (`frontend/tissue.js`) edits organ/lesion/hotspot polygons the
  same way (drag vertices, click an outline to insert, drag inside to move); only
  the layer chosen in *Draw* is hit-testable. Edits invalidate the stale result.

## Recent web/API additions

- **Run controls**: `RunRequest` carries `model` (`hybrid` Darcy bed | `direct`
  connections), `nonlinear` + `max_iter`/`tol` (nonlinear fixed-point cap, threaded
  through `run_hybrid`→`solve_nonlinear`), and `rho_c` (direct connection radius).
  Non-convergence warnings are captured during the solve and returned in
  `summary["notes"]`. The **direct** model (`runner.solve_direct_model` →
  `store.save_direct_result`) has no grid field: it stores a render blob (vessel
  segments + terminal-connection links with flow) served by `GET /projects/{id}/direct`,
  and the frontend Result view draws that (flow-weighted) instead of a heatmap,
  hiding the field picker / tracer / field export. `simulate` MCP tool gained
  `model`/`rho_c`/`max_iter`.
- **Clear run errors**: `runner.validate_run` raises `InputError` (a `ValueError`)
  with actionable messages (empty mask, mask split into disconnected Darcy regions,
  `p_in == p_out`); `jobs._describe_error` surfaces those verbatim and adds a
  traceback tail for unexpected errors. The frontend shows it in a styled box.
- **Viewport** (`frontend/viewport.js`): every canvas fills its stage (HiDPI) with
  zoom/pan/fit. Map coordinates only through `toCanvas`/`toDomain` (workspace,
  `core.js`) or the explorer's `tc`/`td` — never through raw canvas width/height.
  The workspace domain is per project (`setDomain`, from `grid.domain_limit`).
- **Frontend UX**: global undo/redo (`recordHistory` snapshots the editable
  geometry; ⌘/Ctrl+Z), context-aware keyboard shortcuts (Del/Enter/I/A/V/G/R/Esc/1–3),
  a build-version badge (`GET /frontend-version`) that flags a stale tab, three
  tabs (Vessels · Tissue · Result), and a 2-D/3-D view toggle that keeps the project
  (`GET /projects/{id}/view3d`).

- **Tree explorer growth** (`perfusion/treegen.py`, `POST /explore/grow`, MCP
  `grow_trees_in_tissue`): a fractal skeleton (`grow_tree(max_generations=)` /
  `grow_tree_3d`) from placed roots, then optional space filling. `growth`'s
  colonisation/CCO/sampling are **dimension-agnostic** (2-D or 3-D by the base
  network / mask), and `pipeline3d` now snaps terminals outside a 3-D mask to the
  nearest active cell (`_snap_to_active_3d`), like 2-D — keep both invariants.
  `grow_tree_3d` rotates each bifurcation's plane by `twist` relative to its
  parent's (the old fixed-axis rule left every generation in one plane — keep the
  volumetric regression test); optional `gamma`/`exponent` (Murray asymmetry),
  `jitter`, `seed`. `run_hybrid_3d` also returns `flux` + `tof` (masked faces carry
  no flux). Organ shapes live in `perfusion/organs.py` (2-D outlines, 3-D masks);
  every shape must stay **one connected region** (`tests/test_organs.py`).
  `organs.roots(name, dim, domain, size)` gives each organ's **anatomical** artery
  inlet / vein outlet + trunk aim (renal hilum, aortic root, porta hepatis, circle
  of Willis, …); `/explore/grow` and the MCP tool use them whenever a root isn't
  sent, and the explorer follows them until the user moves a root. A new organ
  needs an outline/mask *and* a roots entry whose trunks lead into the tissue.

- **Coupling choice** (`run_hybrid`/`run_hybrid_3d(coupling=)`): `"schur"` (dense
  terminal response, reusable via `darcy_response=`) or `"monolithic"` (bed pressures
  as unknowns, `network.MonolithicDarcy`, `O(nnz)`); `"auto"` switches above
  `MONOLITHIC_AUTO_ENTRIES`. They must agree to machine precision — keep
  `test_monolithic_coupling_matches_schur` green. TOF treats inflow `< 1e-12·max` as
  stagnant (`timeofflight._dead_cells`), else round-off makes it ill-posed.
- **3-D projects**: `GridIn.nz` set ⇒ 3-D (6-value `domain_limit`, raster `(nx,ny,nz)`
  mask, `node_z` trees); `runner.solve` → `run_hybrid_3d`. Grow / prune / adapt,
  tracer (+ kinetics, kinetic maps) and oxygen work on 3-D projects: `HybridResult3D`
  carries `fx/fy/fz` + `sources`, and `transport`/`oxygen` take `fz=`; the viewer shows
  them via `GET /projects/{id}/view3d?field=` (incl. `tracer` per-frame voxels).
  UQ / Sobol / calibration run on 3-D too: `uq._sample_solver` dispatches a `Grid3D` to
  `run_hybrid_3d` with fixed trees and one reused `darcy_coupling_3d` (unless K varies);
  only `uq.PARAMS_3D` (p_in, p_out, mu, permeability) may vary — `r_min_frac`/
  `hematocrit` don't enter the 3-D solve and are rejected, never silently ignored. The
  multicompartment bed is dimension-agnostic (`multicompartment._tpfa/_source`, fields
  keep `grid.mask.shape`; N=1 ≡ `run_hybrid_3d`), shown as `view3d?field=mc_*`.
  **Hand editing** works on 3-D trees too: `PUT /editable` takes `node_z` (required on a
  3-D project, rejected on 2-D), `helpers.rebuild_network(node_z=)` keeps z and 3-D
  lengths (`helpers.segment_lengths`); the viewer's editor is `frontend/edit3d.js`
  (screen-plane drags via `viewer3d.unproject`, which must stay the exact inverse of
  `project`). The **direct-connection model** runs on 3-D projects too:
  `direct.connect_terminals` links terminals by 3-D distance (`Network.positions`;
  mixing a 2-D and a 3-D tree raises), the direct blob / network-tracer lines carry z
  (`Network.segment_lines`: 7-col rows `[x,y,z,x,y,z,flow]`, 6-col tracer lines), and
  `view3d` returns `direct` (+ `net_tracer` for `field=tracer`) instead of a voxel field
  — any other field is a 409. **Field exports** (`export?fmt=csv|vtk`, `/bundle`) work in
  3-D: `io.fields_to_csv/_to_vtk` take a 2-D or 3-D field (4/6-value `domain_limit`;
  CSV `x,y[,z],…` rows x-fastest, VTI a voxel volume); a direct result has no field →
  409 (`_require_field_result`), its vessels still export as `vtp`. The bundle also ships
  `tracer.npz`. **3-D flow videos**: `tracer.gif`/`tracer.mp4` on a 3-D project render
  `helpers.tracer_images_3d` — a maximum-intensity projection of the bolus + the trees,
  seen through `helpers.project_3d` (the same camera as `viewer3d.js`'s `project` — keep
  them in step), turning once round the organ unless `spin=false`. **Lesions & hotspots
  in 3-D** are spheres (`center` [x,y,z] + `radius`; schemas `_Region` = polygon *or*
  ball, discs allowed in 2-D, `PUT` checks the shape matches the project's dimension).
  `geometry.rasterize_regions` rasterises polygons/discs/spheres (later wins);
  `runner.build_grid3d` turns lesions into a per-voxel `Grid3D.permeability` field and
  `darcy3d._face_transmissibilities` uses the harmonic mean per face (a uniform field must
  stay identical to the scalar — tested); hotspots become the 3-D growth `density`. The
  viewer edits them in `frontend/regions3d.js`; the 2-D Tissue editor turns a disc into a
  32-gon on load. Still 2-D-only (`deps.get_project_2d` → 409): drawn tissue masks and
  the raw `GET /tracer` / `results/field` grids. New
  project features must either work in 3-D or be guarded. Oxygen's Damköhler rate is
  scaled by the *active* tissue volume (`oxygen.solve_oxygen`) — keep it that way.
- **Default trees** (no vessels drawn): `runner.default_trees` fits the thesis trees
  jointly into the project domain; every solve path uses
  `build_networks(defaults=True)`; sweeps rebuild them per sample via
  `runner.tree_builder` (so `r_min_frac` stays effective).
- **Run comparison (A/B)**: `POST /projects/{id}/snapshots` keeps a copy of the current
  *field* result blob (`{pid}.snap.{sid}.npz`) + its summary in the `snapshots` table;
  `GET /compare?snapshot=` and `GET /compare/field/{name}?snapshot=` (2-D) / `view3d?field=
  d_<name>&snapshot=` (3-D, symmetric range, `delta: true`) return Δ = current − baseline.
  The maths is `core.compare` (pure NumPy, 2-D/3-D); a direct result has no field to keep
  (409). UI: `frontend/compare.js` (shared panel; diverging ramp in `palette.js`) in the
  2-D Result tab and the 3-D viewer; MCP `compare_scenarios` (same trees, two settings).
- **Browser tests**: `tests/ui/` (Playwright; `--extra ui`), run in the CI `ui` job.

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