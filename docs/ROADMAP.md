---
title: "perfusion — Development Roadmap"
subtitle: "From a Python modelling library to a hosted research & teaching platform"
author: "Planning document"
---

# 1. Vision

Turn the `perfusion` package into a **hosted web application** for modelling blood
circulation and perfusion in organs, usable both for **research** (writing papers,
extending the thesis) and **teaching** (students run and explore simulations).

> **The mathematics** behind every module — Poiseuille/Darcy/TPFA, the hybrid Schur
> coupling, multi-compartment exchange, tracer kinetics & deconvolution, oxygen
> extraction, structural adaptation, morphometry, growth, Windkessel and the UQ/Sobol
> estimators — is written up with equations and thesis citations in
> **[`docs/MODELS.md`](MODELS.md)**. Profiled optimisation opportunities are tracked in
> §10 (Performance & optimisation).

A user should be able to, in the browser:

- load a blood-vessel drawing/image, or **draw tissue and vessels by hand**;
- **segment** an image into a vascular network;
- **grow** new vessels from selected segments;
- configure and **run** a simulation;
- inspect **2-D and 3-D** results and **flow videos**;
- **export** figures, videos and raw results.

Decided constraints (this plan is built around them):

| Decision | Choice |
|---|---|
| Front-end | **Web app** — FastAPI back-end + browser UI (2-D canvas + WebGL/vtk.js 3-D) |
| Dimensionality | **2-D first**, then 3-D as a dedicated phase |
| Audience | **Research and teaching equally** |
| Deployment | **Shareable / hosted** (multi-user sessions, persistence) |

---

# 2. Target architecture

A strict layering keeps the physics reusable and the app thin. Each layer only
depends on the ones below it.

```
┌───────────────────────────────────────────────────────────────┐
│  Web front-end (React/TypeScript)                               │
│  • project workspace • image upload • draw tissue/vessels       │
│  • run controls • 2-D canvas + 3-D vtk.js views • video • export│
└───────────────▲───────────────────────────────────────────────┘
                │  HTTP/JSON + binary (OpenAPI contract), WebSocket for progress
┌───────────────┴───────────────────────────────────────────────┐
│  API layer   perfusion.api  (FastAPI)                           │
│  • projects/sessions • jobs (async) • persistence • schemas     │
└───────────────▲───────────────────────────────────────────────┘
                │  plain Python calls
┌───────────────┴───────────────────────────────────────────────┐
│  Services    segmentation · growth · optimize · io(+export)     │
│              higher-level operations built on the core          │
└───────────────▲───────────────────────────────────────────────┘
                │
┌───────────────┴───────────────────────────────────────────────┐
│  Compute core   perfusion.core   (pure NumPy/SciPy, no I/O/UI)  │
│  parameters·viscosity·geometry·darcy·network·direct·transport·  │
│  tkm·timeofflight·pipeline                                      │
└─────────────────────────────────────────────────────────────────┘
```

Principles:

- **The core stays pure** — no file, network, plotting, or web dependencies; it is
  importable, deterministic and fully unit-tested. Everything else builds on it.
- **Services are stateless functions** taking core objects in and out.
- **The API owns state** (projects, uploaded images, meshes, results) and
  orchestrates long-running work as **jobs**.
- **The front-end holds no physics** — it renders data the API sends and posts
  user edits (drawings, parameters) back.

## 2.1 Hosted-deployment concerns

- **Sessions & projects**: each user has projects; a project bundles geometry,
  parameters, and results. Start with SQLite + a filesystem blob store; the store
  interface stays swappable (→ Postgres + object storage when scaling).
- **Jobs**: simulations can take seconds-to-minutes. Run them as background jobs.
  ✅ *Now:* an in-process thread pool with a **durable SQLite job registry**
  (survives restarts; stale jobs reaped). *Next:* a task queue (RQ/Celery + Redis)
  behind the same store seam, with progress over WebSocket.
- **Statelessness of compute**: the core takes explicit inputs and returns
  results — no globals — so it parallelises across workers cleanly.
- **Auth**: out of scope initially (single lab); design endpoints so an auth
  middleware can be added without touching services.

---

# 3. Feature → component map

Every requested capability, and where it lives.

| Feature | Front-end | API endpoint (sketch) | Back-end |
|---|---|---|---|
| Load vessel drawing/image | ✅ file picker | ✅ `POST /projects/{id}/image` | ✅ `segmentation.segment_network` + `network_in_domain` |
| Draw tissue (mask) by hand | ✅ polygon editor (vertex drag/insert/delete, organ presets) | ✅ `PUT /projects/{id}/mask\|lesions\|hotspots`, `POST /shapes/polygon` | ✅ `rasterize_mask` → `Grid.mask`; `organs.polygon` |
| Draw vessels by hand | ✅ node-graph editor (draw, branch, drag, per-segment radius, inlet/outlet nodes) | ✅ `GET/PUT /projects/{id}/editable` | ✅ `helpers.rebuild_network`; `Network.inlet_nodes` |
| Prune vessels | ✅ steps + undo | ✅ `POST /projects/{id}/prune` | ✅ `geometry.prune_terminal_branches` |
| Grow vessels from selected segments | ✅ method + run | ✅ `POST /projects/{id}/grow?…method=` | ✅ `growth` (space colonisation + CCO) |
| Run simulation | run + progress | `POST /projects/{id}/run` (job) | `core.pipeline.run_hybrid` |
| 2-D result views | ✅ canvas heatmap, zoom/pan, HiDPI | ✅ `GET /projects/{id}/results/field/{name}` | `core` fields → JSON |
| 3-D result views | ✅ dependency-free canvas (rotate/zoom/pan); vtk.js optional | ✅ `GET /projects/{id}/view3d`, `/demo3d`, `POST /explore/grow` (dim=3) | ✅ `pipeline3d` (pressure + flux + TOF); `.vtp` export |
| Grow trees into an organ (2-D/3-D) | ✅ Tree explorer: organ, roots, generations, fill, size | ✅ `POST /explore/grow`, `GET /shapes` | ✅ `treegen`, `organs`, `growth` (2-D/3-D) |
| Flow video (2-D/3-D) | ✅ player + scrubber | ✅ `POST/GET /projects/{id}/tracer`, `…/tracer.gif`, `…/tracer.mp4` | `core.transport` (bolus advection) + GIF / MP4 |
| Export results | ✅ download + format picker | ✅ `GET /projects/{id}/export?fmt=npz\|csv\|vtk` | ✅ `io` exporters (npz/CSV/VTK); MP4 pending |
| Parameter sweeps / experiments | forms | `POST /experiments` (batch jobs) | `optimize`, batch runner |

New back-end pieces flagged **new** are scheduled in the phases below.

---

# 4. Back-end research roadmap

Improvements that make the tool publishable and extend the thesis. Each is
self-contained and testable against the existing physics checks.

**Scalability**

- **Monolithic sparse coupling** — replace the dense `T×T` terminal-response
  (currently `O(T²)` memory) with one sparse saddle-point solve of network +
  Darcy cells. Enables fine grids and thousands of terminals.
- **Iterative + preconditioned solvers** — CG + algebraic multigrid (`pyamg`) for
  the SPD Darcy block; keep a symmetric datum (see review item #7).
- **Optional acceleration** — `numba`/`JAX` for hot kernels; cached factorisations
  for parameter sweeps.

**Physics / methods**

- **Heterogeneous & anisotropic permeability** (per-cell `K`; MPFA on skewed grids).
- **Vessel-wall elasticity** (Eq. 4.9) inside the nonlinear solve.
- **Rheology**: plasma skimming, hematocrit transport, non-Newtonian options.
- **Multi-compartment / dual-porosity** capillary beds.
- **Vessel growth**: constrained constructive optimisation (CCO) and
  space-colonisation, seeded from user-selected segments — directly continues the
  thesis "grow the vascular network" outlook and Paper D optimisation.
- **Inverse problems**: estimate perfusion parameters by fitting the model to
  DCE-MRI curves (ties Paper C to real data); add uncertainty/sensitivity analysis.

**3-D (Phase 4)**

- 3-D structured Darcy grid (reuse TPFA in 3-D); 3-D fractal trees; 3-D segmented
  networks from image stacks / MRA; VTK/glTF export; 3-D flow video.

---

# 5. Phased plan

| Phase | Goal | Key deliverables |
|---|---|---|
| **0. Reorganise** *(this change)* | Maintainable, layered source | `perfusion.core` split out; stable public API; docs updated |
| **1. Back-end API** | Headless service | ✅ FastAPI `perfusion.api`: projects, async jobs, vessels/grow/run/results/field/network/export; ✅ SQLite + blob store; OpenAPI contract; result exporters (`io`) |
| **2. Web front-end** | Usable 2-D app | ✅ zero-build app (`frontend/`, served at `/`): node-graph **vessel editor** + **tissue polygon editor**, organ presets, grow/prune/adapt, run + poll, field heatmaps, **flow video**, analyses, export, save/load, undo/redo, zoom/pan, shareable URLs; Tree explorer; 3-D viewer. *Open:* optional React/Vite rebuild, automated UI tests |
| **3. Research features** | Publishable depth | monolithic sparse solve + AMG (scale), heterogeneous permeability, **vessel growth** from selections, elasticity |
| **4. 3-D** | 3-D modelling + views | 3-D grids/trees/segmentation, vtk.js rendering, 3-D flow video, VTK export |
| **5. Research tooling** | Studies at scale | parameter sweeps, inverse estimation, UQ, reproducible "experiments", batch jobs |

Phases 1 and 2 can proceed against the **2-D** core; 3-D (Phase 4) slots in behind
the same API/front-end because dimensionality is hidden below the service layer.

---

# 6. Source reorganisation (Phase 0)

Target layout — the physics core is isolated; app/service/UI layers sit beside it
and are added over the phases.

```
src/perfusion/
  __init__.py            # curated public API (stable re-exports)
  core/                  # pure compute — no I/O, viz, or web deps
    parameters.py  viscosity.py  geometry.py  darcy.py
    network.py  direct.py  transport.py  tkm.py  timeofflight.py  pipeline.py
  io.py                  # loaders + (future) exporters: npz/csv/vtk/json/mp4
  viz.py                 # matplotlib figures for reports & scripts
  segmentation.py        # image → Network
  optimize.py            # vascular-structure optimisation
  examples/              # runnable scripts (use the public API)
  growth.py              # [Phase 3] vessel growth from seed segments
  api/                   # [Phase 1] FastAPI back-end (projects, jobs, endpoints)
frontend/                # [Phase 2] React/TypeScript web UI
tests/                   # one file per module + shared fixtures
docs/                    # model reference + this roadmap
```

Why: the current flat layout mixes pure numerics with I/O, plotting and (soon)
web code. Isolating `core` means:

- the compute engine has **no optional dependencies** and is trivially reusable by
  the API, batch jobs, notebooks, or another front-end;
- new layers (`api`, `growth`, `frontend`, 3-D) attach **without disturbing** the
  validated physics;
- tests and imports state intent (`perfusion.core.darcy` vs `perfusion.viz`).

**Compatibility**: the public API stays stable — `from perfusion import Grid,
Parameters, run_hybrid, arterial_tree, venous_tree, Network, HybridResult` and the
convenience module handles (`perfusion.geometry`, `perfusion.darcy`, …) keep
working via re-exports in `perfusion/__init__.py`. Only fully-qualified deep
imports move to `perfusion.core.*`.

---

# 7. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Compute | NumPy, SciPy (sparse) | core; add `pyamg` (AMG), optional `numba`/JAX |
| API | **FastAPI** + Pydantic + Uvicorn | async jobs, OpenAPI, WebSocket progress |
| Persistence | SQLite → Postgres; filesystem → object store | swappable store interface |
| Jobs | `BackgroundTasks` → RQ/Celery + Redis | scale to multi-user |
| Front-end | React + TypeScript, Vite | |
| 2-D drawing | HTML canvas / Konva | tissue mask + vessel polylines |
| 2-D fields | Plotly / canvas | heatmaps, overlays |
| 3-D | **vtk.js** (or Three.js) | fields, meshes, streamlines |
| Video | client playback of frames; server `imageio`/ffmpeg mp4 | |
| Export | npz, CSV, JSON, **VTK/.vtu** (ParaView), PNG, MP4 | |
| Packaging | Docker image (API+static front-end) | hosted deployment |

---

# 8. Status & next steps

**Delivered**

- ✅ **Phase 0** — source reorganised into `perfusion.core` + service layers.
- ✅ **`io` exporters** — `save_results` (npz) and `results_summary` (JSON).
- ✅ **`network_from_polylines`** — build a `Network` from hand-drawn/edited vessel
  polylines (shared `network_from_edges` builder, reused by segmentation).
- ✅ **`perfusion.api` skeleton** — FastAPI app (`health`, `projects`, `vessels`,
  `grow`, `run` as an async job, `jobs`, `results`, `field`, `export`); in-process
  job runner; OpenAPI contract generated. Run with `uv run perfusion-api`.
- ✅ **`growth`** — vessel growth by space colonisation (`grow_network`,
  `grow_in_mask`), seeded from selected segment tips, with Murray-law radii;
  exposed as `POST /projects/{id}/grow` (the grown tree feeds the next run).
- ✅ **Phase 2 front-end (v1)** — a **zero-build** 2-D workspace (plain
  HTML/CSS/ES-modules in `frontend/`, served at `/` by the API): create project,
  draw vessels, grow, run (async job + poll), view pressure/flux/TOF as a viridis
  heatmap with the network overlaid, export, and open a project by shareable
  `?project=<id>` URL. Added `GET /projects/{id}/network` for the overlay.
- ✅ **Persistence** — `ProjectStore` is now SQLite (metadata) + a filesystem
  blob store (grown network + result `.npz`), rooted at `$PERFUSION_DATA`.
  Projects and results survive a process restart, so shared URLs keep working;
  the store interface stays swappable (→ Postgres + object storage).
- ✅ **Tissue-mask drawing** — outline the organ on the canvas (Tissue mode);
  `core.geometry.rasterize_mask` turns the polygons into `Grid.mask`, so both the
  Darcy solve and vessel growth are organ-shaped (not the bounding box). Regions
  persist with the project; `PUT/GET /projects/{id}/mask`.
- ✅ **Flow video** — a contrast bolus is advected through the tissue
  (`core.transport`, timed off the time-of-flight); the UI plays/scrubs the
  frames as a 2-D flow animation with a robust (99th-pct) colour scale, and an
  animated GIF can be downloaded. `POST/GET /projects/{id}/tracer` (job) +
  `GET /projects/{id}/tracer.gif` (needs the `image` extra) / `tracer.mp4`
  (H.264 via the `video` extra's bundled ffmpeg).
- ✅ **Image upload + segmentation (artery & vein)** — `POST /projects/{id}/image?kind=artery|vein`
  segments an uploaded vessel image (`segmentation.segment_network`) and places
  it in the project's domain (`segmentation.network_in_domain`) as the arterial
  or venous network; both are persisted, overlaid in the UI (red/blue), and
  coupled on the next run. Needs the `image` extra.

- ✅ **Durable job registry** — jobs (id, project, kind, state, error,
  timestamps) are recorded in SQLite, so status/history survive a restart and
  stale in-flight jobs are marked `interrupted` on startup. `GET /jobs[?project_id=]`
  lists them. The `JobManager` writes lifecycle through the store, which is the
  seam a real task queue slots into.
- ✅ **VTK / CSV export** — `GET /projects/{id}/export?fmt=npz|csv|vtk`: the
  fields as ParaView-readable VTK ImageData (`.vti`) or a long-format CSV (one
  row per active cell), alongside the raw `.npz` (`io.fields_to_vtk/_to_csv`).
  The UI has a format picker next to Export. **3-D projects** export too: CSV
  `x,y,z,…` per active voxel, a VTI voxel volume, the bundle (with `tracer.npz`) and the
  vessels as `.vtp` — from the 3-D viewer's export row (+ 📷 PNG); MCP
  `export_tissue_fields` writes the same files for an organ model.
- ✅ **CCO growth variant** — `growth.cco_grow` (constrained constructive
  optimisation): terminals added one at a time by nearest-segment splitting,
  spaced ≥ `d_min`, Murray radii. Selectable via `POST /grow` `method=cco` and a
  method picker in the UI, beside space colonisation.
- ✅ **Multiple inlets / outlets** — the solver prescribes `p_in`/`p_out` at a
  *set* of boundary nodes (`Network.inlet_nodes`), not just the root; node
  roles are classified by **degree** so any leaf can be an inlet/terminal.
  *(Superseded in the UI by inlet/outlet **nodes** on the editable network — see
  "Interactive network editing" below; `PUT/GET /projects/{id}/io` points remain as a
  legacy path, snapped onto nodes when a project opens.)*

- ✅ **Separate feeding trees (multi-component)** — `network_from_edges`/
  `_polylines` orient each connected component from its own root and seed
  `inlet_nodes` with one root per component, so several disconnected arterial/
  venous trees share one Darcy bed. `Network.node_components()` + the API ensure
  every component keeps ≥1 inlet (auto-topped-up) so no block is singular.
- ✅ **MCP server + dev guidance** — `perfusion.mcp_server` exposes the engine as
  MCP tools; `.mcp.json` + `mcp` extra; `docs/DEVELOPING.md` extension guide.
  Convention: new features also get an MCP tool.
- ✅ **CI** — GitHub Actions runs ruff + the full test suite on Python 3.10/3.12
  and a front-end syntax check (`.github/workflows/ci.yml`).
- ✅ **Full 3-D hybrid coupling (Phase 4)** — `core.darcy3d` (`Grid3D`, 3-D TPFA,
  `point_source_3d`, `terminal_response_3d`, `solve_pressure_3d`, `face_fluxes_3d`),
  3-D fractal trees (`geometry.grow_tree_3d`, `Network.node_z`), and
  `pipeline3d.run_hybrid_3d` couple 3-D arterial/venous trees to a 3-D Darcy bed.
  The network solver is *coordinate-free*, so it is reused verbatim. Verified:
  the coupled 3-D solve conserves mass (arterial inflow == venous outflow), pins
  inlets/outlets, and supports multiple inlets. Demo: `perfusion-hybrid3d`.
- ✅ **3-D viewer in the browser** — `GET /demo3d?model=synthetic|frog` serves a
  result (tree segments + a subsampled voxel field); `frontend/viewer3d.html`
  renders it as a rotatable 3-D scene with a **dependency-free canvas** projection
  (no WebGL/vtk.js — works offline, verifiable). Two models: a synthetic 3-D
  hybrid, and the **frog tongue** shown as a 3-D pressure-relief surface with its
  vasculature at the base. Linked from the 2-D workspace (whose left-panel tools
  are now **contextual to the active mode tab** for a tidier UI). *(vtk.js/WebGL volume rendering could swap in
  behind the same `/demo3d` data later for fancier transfer functions.)*
  *Next 3-D:* 3-D segmentation from image stacks; a full 3-D project path in the API/UI.
- ✅ **Interactive network editing** — the workspace's Vessels tab edits the
  artery/vein networks directly (draw, extend, branch, drag, **per-segment radius**,
  split, join, delete); **inlets/outlets are network nodes** (`inlet_nodes`, `I`
  key) — the separate I/O and Edit tabs are gone, legacy strokes / I/O points are
  converted on open. `PUT /editable` can create or remove a tree.
  `io.save_network` now persists `inlet_nodes`/`node_z` (multiple outlets were
  silently lost on reload before). `validate_run` rejects a tree with no free end.
- ✅ **Tissue polygon editor** — organ / lesion / hotspot layers edited vertex by
  vertex; **organ presets** (`organs.polygon`, `POST /shapes/polygon`); per-project
  **domain size** (Parameters → width/height; grid ny follows the aspect).
- ✅ **Pruning** — `geometry.prune_terminal_branches` removes the outermost
  terminal-branch generations (never the fed trunk); `POST …/prune`, web Grow/prune
  card, MCP `prune_vessel_tree`.
- ✅ **Tree explorer growth (2-D + 3-D)** — `perfusion.treegen` + `POST
  /explore/grow` + MCP `grow_trees_in_tissue`: organ tissue (11 2-D / 10 3-D shapes,
  `perfusion.organs`, all tested connected) with **anatomical artery/vein entries**
  (`organs.roots`), fractal skeleton of N generations (`grow_tree(max_generations=)`),
  space filling by colonisation or CCO, adjustable object size (×0.5–×16) and grid
  resolution, solve + homogeneity objective, animate-growth replay.
- ✅ **Truly volumetric 3-D trees** — `grow_tree_3d` twists each branching plane
  relative to its parent's (the old fixed-axis rule kept every generation in one
  plane); Murray-law asymmetry, seeded jitter.
- ✅ **3-D flux + time-of-flight** — `run_hybrid_3d` returns `flux` and `tof`
  (`darcy3d.face_fluxes_3d` masked, `flux_magnitude_3d`, `upwind_operator_3d`,
  `timeofflight.time_of_flight_3d`), and 3-D terminals outside the mask snap to the
  tissue (`_snap_to_active_3d`). The explorer computes fields and objectives in 3-D.
- ✅ **Viewport** — every canvas fills the window, HiDPI-crisp, with zoom about the
  cursor, pan and fit (`frontend/viewport.js`), in the workspace, explorer and 3-D
  viewer.

- ✅ **Task queue (optional RQ)** — job execution moved to `api.runner` +
  `run_job` (reconstructs the store from its root), so it runs in either the
  default thread pool **or** an RQ/Redis worker. `RQJobManager` behind the same
  `submit`/`get` seam; selected with `PERFUSION_QUEUE=rq` (+ `rq` extra, a running
  Redis, and `rq worker`), with graceful fallback to threads. Verified in-process
  with fakeredis.

**Next** (see §11 for the full list of what is still open)

1. A **3-D project path**: store/edit/solve 3-D trees + organ masks in projects (today
   3-D lives in the explorer and the 3-D viewer); 3-D segmentation from image stacks.
2. Scale: **monolithic sparse coupling** instead of the dense `T×T` terminal response
   (needed for fine 3-D grids and thousands of terminals); response reuse in sweeps.
3. Server-side **mp4**; automated UI tests; optional React/Vite rebuild + vtk.js.

Each step is independently useful and testable; the API contract lets the
front-end be built in parallel.

# 9. Research & algorithm backlog

Directions to grow the science and the platform, roughly ordered by value ÷ effort
within each group. Effort: **S** ≲ a day, **M** a few days, **L** a week+, **XL**
a research project. Each notes where it lands in the tree.

## 9.1 Physics & modelling

| Idea | What it adds | Lands in | Effort |
|---|---|---|---|
| **Transit-time distribution & CTH** *(in progress — see 9.6)* | turn the TOF field into MTT, capillary transit-time heterogeneity (σ) and an oxygen-extraction estimate | `core/perfusion_metrics.py` | S |
| ✅ Vessel-wall elasticity (thesis Eq. 4.9) | pressure-dependent radii, r = r0(1 + C(p − p_out)); fixed-point in `run_hybrid(elasticity=True)` + `Parameters.compliance` | `pipeline` | done |
| Pulsatile / transient flow | cardiac-cycle dynamics via RLC / Windkessel + inertance | new `core/transient.py` | L |
| ✅ Hematocrit phase-separation at bifurcations (Pries plasma-skimming) | unequal RBC split → per-segment hematocrit & viscosity vary (`core/hematocrit.py`, coupled in `run_hybrid(phase_separation=True)`) | `core/hematocrit.py` + `pipeline` | done |
| ✅ Multi-compartment Darcy (arteriole/capillary/venule beds) | N stacked porous compartments coupled by inter-compartment exchange β (Michler/Hyde–Smith); a generalised terminal response plugs into the unchanged network solver, N=1 reduces to `run_hybrid` (`core/multicompartment.py`, `run_multicompartment`, web card + MCP `multicompartment_perfusion`) | `core/multicompartment` + `pipeline` | done |
| ✅ Oxygen / solute transport + extraction | steady advection–reaction(–diffusion) of a blood-borne solute (`core/oxygen.py`): delivery at feeds, first-order tissue uptake at Damköhler rate k, venous washout → saturation / uptake maps + global OEF | `core/oxygen.py` | done |
| ✅ Heterogeneous permeability | per-cell `K(x)` field + harmonic-mean transmissibilities (`Grid.permeability` scalar-or-field, `rasterize_permeability`, MCP `heterogeneous_perfusion_lesion`) + an interactive low-K-lesion painting UI in Tissue mode + **diagonal-tensor anisotropy** (`Kx ≠ Ky`, axis-split harmonic transmissibilities, `GridIn.permeability_y` + a web K-anisotropy input, MCP `anisotropic_perfusion_bed`); *todo:* full (off-diagonal) tensor K via MPFA | `core/darcy` + `core/geometry` + web | done |
| ✅ Structural adaptation / autoregulation (Pries–Secomb) | wall-shear-driven radius remodelling drives the tree toward Murray's law (`|Q| ∝ r³`), early-stopping at the target exponent (`core/adaptation.py`, `run_adaptation`, `POST …/adapt`, web Adapt card, MCP `adapt_vessel_network`) | `core/adaptation` + `pipeline` | done |
| Poroelasticity (Biot) | tissue deformation ↔ perfusion coupling | new module | XL |

## 9.2 Numerics & solvers

- ◑ **Newton with analytic Jacobian** for the nonlinear solve — *deliberately not added.*
  The fixed-point (`solve_nonlinear`) reuses one LU factorisation, reduces exactly to
  `solve_linear` when junction losses are off, and is ~12× faster than the old dense-Jacobian
  Newton that was removed; for the mild junction-loss nonlinearity a re-added Newton would
  regress speed for marginal robustness gain. Revisit only if a stiffer nonlinearity (e.g.
  strong elasticity) makes the fixed point diverge.
- ✅ **AMG preconditioning** (`core/linsolve.py`, optional `pyamg` extra) — `solve_spd(method=
  "amg")` builds a smoothed-aggregation hierarchy + CG (near-linear scaling for large/3-D
  grids); `solve_pressure_3d(solver="amg")`. Matches the direct solve to ~1e-9; the benchmark
  harness compares the two.
- ◑ **MPFA-O flux** for full (off-diagonal) tensor / non-K-orthogonal media — *deferred
  (genuine numerics research).* Diagonal-tensor anisotropy (`Kx ≠ Ky`) is done and TPFA-exact
  for it (see 9.1); a correct O-method (vertex interaction regions, local flux/pressure
  continuity) is a substantial, separately-validated addition — not worth shipping unverified.
- ◑ **GPU / matrix-free** (cupy) for `darcy3d` — *deferred (needs GPU hardware to implement and
  verify honestly).* `solve_spd` isolates the linear solve, so a cupy backend can slot in later.
- ✅ **Uncertainty quantification** — Latin-hypercube Monte-Carlo over uncertain inputs
  (`uq.monte_carlo`) with QoI mean/std/percentiles + first-order Pearson sensitivity, *and*
  variance-based **Sobol indices** (`uq.sobol_indices`) via a polynomial-chaos surrogate
  (first-order S₁ + total-effect Sₜ, with the fit R²). `POST …/uq?method=correlation|sobol`,
  a web UQ card (Pearson bars or S₁/Sₜ bars) and MCP `uncertainty_quantification` /
  `sobol_sensitivity_indices`.

## 9.3 Growth & geometry

- ✅ **3-D CCO and 3-D space colonisation** — `growth` sampling, colonisation and CCO are
  dimension-agnostic (2-D or 3-D by the base network / mask); tested in 3-D and used by the
  explorer. ◑ *Staged / optimised CCO* (global re-optimisation of bifurcation positions)
  remains deferred; structural adaptation already optimises the Murray exponent (9.1).
- ◑ Loopy venous networks / anastomoses — *partially covered.* The **solver** already handles
  cyclic venous graphs (segmented veins contain loops); only the *growth* algorithms are
  tree-only. Growing anastomoses is a research effort (loop-closure heuristics) — deferred.
- ✅ **Data-constrained growth** — attraction points sampled ∝ a target perfusion-density
  map (`growth.sample_points_in_mask(density=)`, threaded through `grow_in_mask`/
  `cco_grow_in_mask`), so the tree grows denser where demand is high. Painted as **growth
  hotspots** in the web Tissue tab (draw-target toggle + demand weight, teal overlay, stored
  per project + document round-trip, `GET/PUT …/hotspots`, rasterised to a density map by the
  grow endpoint) and MCP `data_constrained_tree_growth`. *(Follow-up: match Strahler-order
  statistics; the demand bias is strongest from a sparse base — a full pre-grown tree dilutes
  it.)*
- ✅ **Strahler-order morphometry** (`core/morphometry.py`) — assigns Horton–Strahler orders
  to a tree and reports the bifurcation (R_B), length (R_L) and diameter (R_D) ratios + per-
  order segment counts/means; exact R_B=2 on a complete binary tree, ~2.1 on the fractal tree.
  `GET …/morphometry`, a web Morphometry card (ratios + per-order table) and MCP
  `strahler_morphometry`. Gives grown / data-constrained trees a morphometric fingerprint to
  match measured vasculature.
- ✅ **Domain-aware growth** (`growth.mask_without_obstacles`, `grow_in_mask(obstacles=)`) —
  tissue-mask holes already give organ-shaped, hole-avoiding growth; an explicit ``obstacles``
  list of no-grow polygons steers both attraction points and terminals around sub-structures.

## 9.4 Tracer kinetics & perfusion analysis

- ✅ **Network-transit kinetics for the direct model** — `core/network_transit.py` times a
  bolus along the vasculature (`τ = volume/|flow|` per element, accumulated arterial →
  capillary connection → venous via a pressure-ordered, loop-robust sweep from the inlet)
  and produces a per-element animation. The field-less direct model now animates the bolus
  on the network: `POST /tracer` on a direct result runs `run_network_tracer`,
  `GET …/network-tracer` serves the element lines + concentration frames, and the ▶ Tracer
  button sweeps the viridis bolus across the vessels (MCP `direct_transit_metrics` reports
  the transit statistics).
- ✅ **DCE/DSC-MRI tracer-kinetic fitting** — `core/tkm.py` fits MS/1CM/2CUM/2CXM
  (`Fp, vp, PS, ve`); now wired to the app: the tracer stores its AIF, and
  `GET /projects/{id}/kinetics` fits the model to the tissue curve (result-view card +
  MCP `fit_tracer_kinetics` tool).
- ✅ **Voxel-wise parametric maps** — `tkm.kinetic_maps(frames, times, aif, mask)` turns
  the tracer time-series into per-voxel **Fp** (max-slope `maxₜ dC/dt / max(Cₐ)`) and
  **vp/blood-volume** (AUC ratio `∫C/∫Cₐ`) maps, NaN on inactive voxels;
  `GET /projects/{id}/kinetics/maps?field=Fp|vp` returns the field and the TKM card's
  **Fp map / BV map** buttons render it on the main heatmap.
- ✅ **Residue-deconvolution maps** — `tkm.deconvolve` (truncated-SVD, Østergaard) and
  `tkm.deconvolution_maps` recover the flow-scaled residue `k = Fp·R` per voxel and read
  off **Fp** (residue peak), **CBV** (∫k) and **MTT** (CBV/Fp) via the central-volume
  theorem. Exposed as `GET …/kinetics/maps?method=deconv&field=Fp|CBV|MTT` (+`oi`), a
  **SVD deconvolution** toggle / **MTT map** button in the TKM card, and the MCP
  `perfusion_parameter_maps(method="deconv")` tool. Validated to ~15 % on a well-sampled
  synthetic 1CM curve; absolute accuracy needs the transit time well-resolved (more
  `n_frames`), while relative rCBF/rCBV/rMTT are robust — the clinical norm.
- ✅ **Block-circulant oSVD** (`tkm.deconvolve(circulant=True)`, `…/kinetics/maps?osvd=true`) —
  the delay-insensitive deconvolution variant: a bolus arrival lag between AIF and tissue no
  longer biases the recovered flow (validated: oSVD Fp shift ≈0 vs sSVD's under a lag).
  *(Per-voxel model fitting remains a follow-up.)*
- ✅ **Transit-time distributions & CTH** → MTT, CTH and oxygen-extraction limits
  (Jespersen–Østergaard), in `core/perfusion_metrics.py` — see 9.6.

## 9.5 Optimisation, inverse problems, platform

- ✅ **Optimisation driver** over `optimize.objective` (Paper D flux-vs-TOF) —
  `optimize.optimize_tree` (Nelder-Mead / differential-evolution over the tree's
  branching asymmetries), exposed as the MCP `optimize_tree_config` tool.
- ✅ **Calibration / inverse recovery** (`inverse.calibrate`) — bounded 1-D search over
  forward solves recovers permeability / inlet pressure / viscosity from a measured QoI
  (arterial inflow, mean flux, TOF); `POST …/calibrate`, MCP `calibrate_parameter_to_flow`.
  *(Adjoint gradients for high-dimensional inversion remain a research follow-up.)*
- ✅ **Property-based conservation tests** (`tests/test_properties.py`, hypothesis) — mass
  conservation asserted across the random parameter space for the hybrid + multi-compartment
  solvers.
- ✅ **Network VTK `.vtp` export** (`io.network_to_vtp`, `GET …/export?fmt=vtp`) — vessels as
  ParaView PolyData tubes (radius / Strahler order / flow cell data).
- ✅ **Benchmark harness** (`scripts/benchmark.py`) — times `run_hybrid` across grid sizes and
  compares the direct vs. AMG Darcy backends. *(A full asv cross-commit harness is optional.)*
- ◑ **Frontend automated tests** — every web feature this cycle was smoke-verified with a
  headless-Chrome CDP driver; codifying it as a Playwright/CI job (needs a JS toolchain + a
  server+Chrome in CI) is the remaining step. Deferred.
- ◑ **Run-comparison view / parameter-sweep UI** — *partially covered.* The **Tree Explorer**
  page gives interactive parameter exploration with a live render, and MCP `run_sweep` sweeps a
  parameter; a dedicated in-app A/B result-comparison needs a multi-result-per-project data
  model. Deferred.
- ◑ **Units safety (pint)** — *deferred by design.* The engine is deliberately non-dimensional
  (thesis convention); wrapping every quantity in pint units is a pervasive change with little
  benefit for the research use-case.

## 9.6 Delivered from this backlog

- ✅ **Transit-time distribution & CTH** (`core/perfusion_metrics.py`): MTT, CTH (σ of the
  transit-time distribution), relative dispersion RD = CTH/MTT, and a Kety-model oxygen
  extraction fraction over the empirical distribution — surfaced in the result summary,
  a `GET /projects/{id}/transit` endpoint + an h(τ) chart in the result view, and the MCP
  `simulate` output.
- ✅ **Optimisation driver** (`optimize.optimize_tree`) — minimise flux-/TOF-homogeneity
  over the tree's branching asymmetries (Nelder-Mead or differential evolution); MCP
  `optimize_tree_config` tool.
- ✅ **Tree Explorer** page (`explorer.html`, `POST /explore` + `/explore/optimize`) —
  interactive fractal-tree parameter sliders with a live field/tree render and a
  one-click optimiser; reachable from the workspace topbar.
- ✅ **Tracer-kinetic fitting** wired to the tracer — AIF stored with the tracer,
  `GET /projects/{id}/kinetics` fits a TKM to the tissue curve, result-view card
  (measured vs fitted curve + Fp/vp/PS/ve), MCP `fit_tracer_kinetics` tool.
- ✅ **Hematocrit phase separation** (`core/hematocrit.py`) — Pries plasma-skimming law
  propagated through the arterial tree; coupled to Fåhræus–Lindqvist viscosity via
  `run_hybrid(phase_separation=True)` (fixed-point), a `RunRequest.phase_separation`
  flag and a Run-card toggle.
- ✅ **Vessel-wall elasticity** (`Parameters.compliance`, `run_hybrid(elasticity=True)`) —
  radii swell with transmural pressure (r = r0(1 + C(p − p_out))), solved as a
  fixed-point; `RunRequest.elasticity`/`compliance` + a Run-card toggle.
- ✅ **Transient / pulsatile (0-D Windkessel)** (`core/transient.py`) — steady R +
  elastic compliance C driven by a pulsatile inlet; `GET /projects/{id}/transient`
  returns the cycle + inlet-vs-tissue pulsatility (damping), a result-view chart,
  and the MCP `pulsatile_windkessel` tool. Default `perfusion-api` port is now 8010
  (`PERFUSION_PORT` to override).
- ✅ **Voxel-wise perfusion maps** (`core/tkm.py`) — semi-quantitative (max-slope Fp,
  AUC blood volume) and residue-**deconvolution** (truncated-SVD Fp/CBV/MTT) maps from
  the tracer; `GET /projects/{id}/kinetics/maps?method=&field=`, TKM-card buttons + SVD
  toggle, MCP `perfusion_parameter_maps`.
- ✅ **Network-transit kinetics for the direct model** (`core/network_transit.py`) — bolus
  timed along the vasculature (τ = volume/flow, pressure-ordered loop-robust sweep);
  `POST /tracer` on a direct result → `GET /projects/{id}/network-tracer`, the ▶ Tracer
  button animates the bolus over the vessels, MCP `direct_transit_metrics`.
- ✅ **Uncertainty quantification** (`uq.py`) — a service layer over the engine:
  Latin-hypercube Monte-Carlo samples uncertain inputs (`p_in`/`p_out`/`mu`/`r_min_frac`/
  `hematocrit`/`permeability`) over given ranges, re-solves, and reports each quantity of
  interest's mean/std/5–95 percentiles + a first-order sensitivity (Pearson r of each input
  vs. output). `POST /projects/{id}/uq`, a web UQ card with per-QoI mean±std and signed
  sensitivity bars, and MCP `uncertainty_quantification`.
- ✅ **Variance-based Sobol sensitivity / polynomial chaos** (`uq.sobol_indices`) — fits an
  orthonormal-Legendre degree-2 PCE surrogate to a modest LHS sample and reads Sobol indices
  off the coefficients analytically: first-order ``S1`` (an input's own variance share) and
  total-effect ``ST`` (incl. interactions, ``ST ≥ S1``), plus the surrogate fit ``R²``. Far
  fewer solves than raw Saltelli MC and constraint-satisfying on this smooth model.
  `POST …/uq` with ``method="sobol"``, the UQ card's Sobol toggle (nested S₁/Sₜ bars), MCP
  `sobol_sensitivity_indices`.
- ✅ **Structural adaptation / autoregulation** (`core/adaptation.py`, `pipeline.run_adaptation`) —
  a reduced Pries–Secomb model: wall shear ``τ_w = 4μ|Q|/(πr³)`` drives radius remodelling
  ``r ← r·exp(rate·S)`` toward a shear set-point. Iterated with a flow solve in the loop the
  tree self-organises toward **Murray's law** (``|Q| ∝ r³``); since the pure-shear stimulus
  over-differentiates past it, the loop early-stops at the closest approach to the target
  exponent and returns that Murray-optimal network (deterministic, mass-conserving).
  `POST /projects/{id}/adapt` remodels + stores the trees, a web Adapt card reports the
  exponent shift, and MCP `adapt_vessel_network` exposes it. *(Follow-up: full Pries–Secomb
  pressure/metabolic/conducted terms for a stable non-Murray equilibrium.)*
- ✅ **Multi-compartment Darcy** (`core/multicompartment.py`, `pipeline.run_multicompartment`) —
  N stacked porous compartments (arteriole → capillary → venule) coupled by a linear
  pressure-exchange β; arterial terminals feed compartment 0, venous drain compartment N−1,
  and the flow percolates down the stack — the inter-compartment exchange field is the
  capillary perfusion. A generalised `terminal_response_multi` places arterial/venous sources
  in the end compartments and returns the same terminal response matrix the **unchanged**
  network solver consumes, so N=1 reduces exactly to `run_hybrid`. Mass is conserved
  (arterial in = venous out = throughflow across every interface). `GET …/multicompartment`
  (perfusion + compartment-pressure fields), a Multi-compartment result card, MCP
  `multicompartment_perfusion`. *(Follow-up: per-compartment heterogeneous K + spatially
  varying β; nonlinear/elastic composition.)*
- ✅ **Heterogeneous permeability** (`core/darcy.py`, `core/geometry.rasterize_permeability`) —
  `Grid.permeability` accepts a scalar *or* a per-cell `(nx, ny)` field; `transmissibilities`
  uses the **harmonic mean** of adjacent-cell K (Eq. 5.1), so a low-K lesion throttles its
  boundary faces and starves local perfusion (ischemia). Scalar K flows through the API
  (`GridIn.permeability`); low-K **lesions** are drawn in the web Tissue tab (draw-target
  toggle + per-lesion k, amber overlay), stored per project (`GET/PUT …/lesions`, document
  round-trip) and rasterised into the grid's K field by `build_grid`; MCP
  `heterogeneous_perfusion_lesion` reports the in-vs-outside flux drop.
- ✅ **Anisotropic (diagonal-tensor) permeability** (`core/darcy.py`) — `Grid.permeability_y`
  gives the capillary bed a distinct y-permeability; `transmissibilities` splits by axis
  (x-faces use `Kx`, y-faces `Ky`, both harmonic-averaged) so TPFA stays consistent for the
  axis-aligned tensor. Throughflow scales exactly with the aligned-axis K (validated), the
  high-K axis channels the flow. `GridIn.permeability_y` (document round-trip), a web
  "K aniso Ky/Kx" input, and MCP `anisotropic_perfusion_bed`. *(Follow-up: full off-diagonal
  tensor K via MPFA-O.)*
- ✅ **Oxygen / solute transport + extraction** (`core/oxygen.py`) — steady
  advection–reaction(–diffusion) of a blood-borne solute: arterial delivery, first-order
  tissue uptake at dimensionless Damköhler rate k, venous washout; a non-singular
  M-matrix solve (C ≥ 0, mass-conservative). Global OEF sits below the well-mixed Kety
  limit 1−e^{−k} exactly as transit heterogeneity predicts. `GET /projects/{id}/oxygen`
  (saturation / uptake maps + OEF), an Oxygen result card, MCP `oxygen_extraction_map`.

# 10. Performance & optimisation

A profiling audit of the compute paths (`src/perfusion/`). The **dominant cost of a
forward solve is `darcy.terminal_response`** — it assembles the TPFA operator, does one
LU (`splu`) of the `n_cells × n_cells` system and back-substitutes one RHS per terminal.
It depends **only** on the grid, the terminal *positions* and `eps` — never on segment
radii or the flow boundary conditions — so any loop that holds the geometry fixed can
reuse it. The findings below are ranked by impact ÷ risk; all are numerically identical
to the previous code (verified to ~1e-12).

## 10.1 Delivered

- ✅ **Reuse the Darcy response across radius-only iterations** (`pipeline.run_hybrid`
  gained `darcy_response=`; `run_adaptation` computes `terminal_response` once and reuses
  it every iteration). Structural adaptation changes only radii, so the Darcy factorisation
  was being rebuilt ~14–60× for nothing. **≈6.3× faster per solve** (140×160: 129 ms → 20 ms);
  byte-identical results. The same hook now lets any outer loop pass a precomputed response.
- ✅ **Drop the redundant operator rebuild in `run_hybrid`** — `pipeline.py:202` reassembled
  the full 5-diagonal `tpfa_operator` just to read `tx, ty`; replaced with the cheap
  `darcy.transmissibilities(grid)` (what `tpfa_operator` calls internally). No numerical change.

## 10.2 Opportunities (measured, not yet applied)

- ✅ *(delivered — see §11.1 #3)* **UQ / calibration / sweep loops reuse the Darcy response** (`uq.monte_carlo`,
  `uq.sobol_indices`, `inverse.calibrate`, `mcp_server.sweep`). When the swept input is
  `p_in`/`p_out`/`mu`/`hematocrit` (grid + terminals fixed) the response is identical across
  every sample, yet recomputed each time — the same ~6× per-solve win as adaptation. Needs a
  guard: the reuse is **invalid** when `permeability` varies (changes the operator) or
  `r_min_frac` varies with fractal trees (moves the terminals). *Medium impact, low risk with
  the guard.*
- **Avoid the `.tolil()` pin round-trip** in `darcy.terminal_response`, `multicompartment.
  terminal_response_multi` and `darcy3d` (2 sites). Converting the whole operator to LIL just
  to zero one Dirichlet-pin row allocates a Python list-of-lists over all nonzeros; the pin can
  be applied in the COO triplets before `tocsc()` or via a CSR indptr edit. *Medium impact on
  large / 3-D grids, low risk.*
- **KD-tree the terminal snapping / de-duplication** (`pipeline._snap_to_active` is
  O(T·n_active) Python; `_spread_coincident` is an O(radius²) spiral search). Build one
  `cKDTree` over active cells and batch-query (as `direct.py` / `growth.py` already do).
  *Low/medium impact (masked grids with many terminals), low risk — nearest cell unchanged.*
- **Vectorise `network_transit.arrival_times` edge construction** (the per-segment orientation
  + weight loop; the arrival *relaxation* is an inherently sequential DAG sweep and must stay a
  loop). *Low impact (direct-model animation only).*
- **Maintain the CCO node/child arrays incrementally** (`growth.cco_grow` rebuilds `arr` and
  the child list from Python lists every terminal → O(n²)). The greedy insertion is inherently
  sequential, but the rebuild is avoidable. *Low/medium impact for large `n_terminals`.*

## 10.3 Correctly sequential — do **not** vectorise

The nonlinear/elasticity/phase-separation fixed points and time-stepping (`solve_nonlinear`,
`run_hybrid` elasticity & phase-sep loops, `transport_forward`, `windkessel`) genuinely change
the operator or state each step and already reuse a single factorisation where possible;
Monte-Carlo/Sobol/sweep *samples* are independent (parallelisable, not vectorisable) and their
win is response-reuse (above), not loop fusion; CCO and space-colonisation growth are iterative
by construction.

---

# 11. What's still open

One consolidated list of everything not yet delivered, gathered from §4–§10 and from
gaps noticed while building the editors / explorer. Ordered by value ÷ effort.
Effort: **S** ≲ a day, **M** a few days, **L** a week+, **XL** research.

## 11.1 Highest value next

| # | Item | Why | Effort | Lands in |
|---|---|---|---|---|
| 1 | ✅ **Monolithic sparse coupling** — `run_hybrid`/`run_hybrid_3d(coupling=)`, `network.MonolithicDarcy`, `darcy[3d].coupling_operator[_3d]`; identical to Schur to machine precision; 48³ × 2 048 terminals: 3.8 → 1.4 GB, 37.7 → 13.5 s; explorer 3-D cap 36³ → 48³ | done | | |
| 2 | ✅ **3-D project path** — `GridIn.nz`/6-value domain, 3-D raster mask + trees, `run_hybrid_3d` in projects, `POST /explore/project` (2-D → workspace, 3-D → viewer with Run + field picker), clear 409s for 2-D-only analyses, document round trip | done | | |
| 3 | ✅ **Darcy-response reuse in UQ / Sobol / calibration / sweeps** — `uq._sample_solver` + `geometry_fixed` guard; 16-sample MC on 120×140: 4.9× faster, identical | done | | |
| 4 | ✅ **Default trees follow the project domain** — `runner.default_trees` (joint fit, `geometry.transform_network`), shown dashed in Vessels and on the result, rebuilt per sample in sweeps | done | | |
| 5 | ✅ **Automated UI tests** — `tests/ui/` (Playwright, real server + Chromium), CI `ui` job | done | | |

- ✅ **Analyses on 3-D projects** — grow (CCO / colonisation in the 3-D mask), prune,
  Murray adaptation (`run_adaptation` takes a `Grid3D`; `pipeline3d.darcy_coupling_3d`
  is reused across iterations), tracer transport (`transport_forward(fz=)`), kinetics
  + Fp/vp/CBV/MTT maps (N-D `tkm`), oxygen (`solve_oxygen(fz=)`); all in the 3-D viewer
  (`view3d?field=`), and the MCP tool `tissue_perfusion_analysis` (2-D or 3-D). Also
  fixed: oxygen's Damköhler rate now uses the *active* tissue volume (a masked organ
  had a diluted rate → OEF ≈ 0.07 instead of ≈ 0.5 at k = 1).
- ✅ **UQ, calibration and multi-compartment on 3-D projects** — `uq._sample_solver`
  runs a `Grid3D` through `run_hybrid_3d` on the fixed 3-D trees, reusing one
  `darcy_coupling_3d` unless K varies (so a 3-D sweep costs ~one LU); inputs limited to
  p_in / p_out / μ / K (`uq.PARAMS_3D`, others → clear 400). `core.multicompartment`
  is dimension-agnostic (N=1 ≡ `run_hybrid_3d`, mass conserved across every interface).
  3-D viewer: `mc_perfusion` / `mc_arteriole` / `mc_venule` fields and an
  *Uncertainty & calibration* card; MCP `tissue_uncertainty_calibration` and
  `tissue_perfusion_analysis(compartments=)`.

Editing 3-D trees by hand is now done too (§11.4).

## 11.2 Modelling & numerics

- **Non-Newtonian rheology options** (shear-thinning apparent viscosity beyond
  Fåhræus–Lindqvist) — §4. *M.*
- **MPFA-O** for full (off-diagonal) tensor permeability / skewed grids — §9.2, deferred. *L.*
- **Staged / optimised CCO** (global re-optimisation of bifurcation positions) — §9.3. *L.*
- **Growing anastomoses / loopy venous growth** (loop-closure heuristics; the solver
  already handles loops) — §9.3. *XL.*
- **Data-constrained growth matched to Strahler statistics** (follow-up in §9.3). *M.*
- **Inverse estimation from real DCE-MRI curves** beyond the 1-D calibration
  (`inverse.calibrate`) — multi-parameter fit of the hybrid model. *L.*
- **GPU / matrix-free `darcy3d`** (cupy) — needs hardware; deferred. *L.*
- ◑ Newton with an analytic Jacobian — *deliberately not added* (the factorisation-reusing
  fixed point is faster and reduces exactly to the linear solve). Units via pint —
  *deferred by design* (non-dimensional engine).

## 11.3 Geometry & data

- **3-D segmentation from image stacks / MRA** into `node_z` networks — §4, §8. *L.*
- **Organ library refinement** — the 3-D heart is a stylised ventricular wall; image- or
  atlas-derived organ masks would replace the analytic shapes. *M.*
- ✅ ~~**Explorer Optimise for the current setup**~~ — `treegen.optimize` (Nelder-Mead /
  differential evolution over γ_a, γ_v, length:radius, Murray exponent, vein calibre,
  3-D twist) on the chosen organ, roots, fill and generations, 2-D or 3-D;
  `POST /explore/optimize` searches on a capped grid, re-scores at the user's resolution
  and never returns a worse setup; **📤 Use optimised trees in simulation** hands it to
  the workspace / 3-D viewer; MCP `optimize_trees_in_tissue`. (The thesis-frame
  `optimize.optimize_tree` / MCP `optimize_tree_config` remain for Paper D.)

## 11.4 Platform & UI

- **Batch experiments / run comparison** — `POST /experiments` (§3), an A/B
  result-comparison view (needs a multi-result-per-project data model) — §9.5. *M–L.*
- ✅ ~~**Server-side MP4** export of the flow video~~ — `GET …/tracer.mp4` (`video` extra:
  imageio-ffmpeg bundles ffmpeg), ↓ MP4 next to ↓ GIF, also in the export bundle.
- ✅ ~~**3-D flow video**~~ — tracer transport on the voxel bed, played back in the
  3-D viewer (`view3d?field=tracer`). An MP4 of the 3-D view is not exported yet.
- ✅ ~~**Workspace result analyses for 3-D**~~ — tracer, kinetics, oxygen, grow, adapt,
  UQ / Sobol / calibration, the multi-compartment bed and now the **direct-connection
  model** (3-D terminal links, flow-weighted links + network bolus in the 3-D viewer, MCP
  `direct_model_in_tissue`).
- ✅ ~~**Hand-editing 3-D trees**~~ — the 3-D viewer's *✏️ edit the trees by hand*
  (`frontend/edit3d.js`): select nodes / vessels, drag nodes in the screen plane (rotate
  to reach depth), Shift+click to branch, split, inlet/outlet toggles, delete, undo;
  saved through `PUT /editable` with `node_z`.
- Optional **React/Vite** rebuild and **vtk.js/WebGL** volume rendering behind the same
  API data — §5, §8. *L.*

## 11.5 Performance (from §10.2, measured, not yet applied)

- ✅ ~~Avoid the `.tolil()` pin round-trip~~ in `darcy.terminal_response` and `darcy3d`
  (`darcy.pin_row`) — now also in `multicompartment`, `darcy3d.solve_pressure_3d` and
  the monolithic `W_src` (no LIL left in `core`).
- ✅ ~~KD-tree the 2-D terminal snapping~~ — `pipeline._snap_to_active` uses `cKDTree`
  (ties → lowest active index, bit-identical to the brute argmin; ~5× on 3 000 terminals).
  The coincident-cell spiral search (`_spread_coincident`) only runs for duplicates and
  is left as is.
- ✅ ~~Maintain CCO node/child arrays incrementally~~ — `growth.cco_grow` keeps
  preallocated position/parent arrays (no per-step list→array rebuild); identical trees,
  ~3× faster. Each step is still one vectorised `O(n)` nearest-segment scan.
- ✅ ~~Vectorise `network_transit.arrival_times` edge construction~~ — edges are built as
  arrays and relaxed through a CSR adjacency with `np.minimum.at`; identical arrival times.
