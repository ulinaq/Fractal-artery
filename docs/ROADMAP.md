---
title: "perfusion — Development Roadmap"
subtitle: "From a Python modelling library to a hosted research & teaching platform"
author: "Planning document"
---

# 1. Vision

Turn the `perfusion` package into a **hosted web application** for modelling blood
circulation and perfusion in organs, usable both for **research** (writing papers,
extending the thesis) and **teaching** (students run and explore simulations).

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
| Draw tissue (mask) by hand | paint canvas | `PUT /projects/{id}/tissue` | rasterise → `core.darcy.Grid.mask` |
| Draw vessels by hand | polyline canvas | `PUT /projects/{id}/vessels` | **new** `core.geometry.network_from_polylines` |
| Grow vessels from selected segments | ✅ method + run | ✅ `POST /projects/{id}/grow?…method=` | ✅ `growth` (space colonisation + CCO) |
| Run simulation | run + progress | `POST /projects/{id}/run` (job) | `core.pipeline.run_hybrid` |
| 2-D result views | canvas/Plotly | `GET /results/{id}/field/{name}` | `core` fields → JSON/PNG |
| 3-D result views | vtk.js | `GET /results/{id}/mesh` | **new** VTK/glTF export (Phase 4) |
| Flow video (2-D/3-D) | ✅ player + scrubber | ✅ `POST/GET /projects/{id}/tracer`, `…/tracer.gif` | `core.transport` (bolus advection) + GIF |
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
| **2. Web front-end** | Usable 2-D app | ✅ *v1 shipped* — zero-build workspace (`frontend/`, served at `/`): vessel **drawing**, grow, run + poll, 2-D field heatmap + network overlay, export, shareable URLs. *Next:* tissue-mask drawing, **flow video**, optional React/Vite rebuild |
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
  `GET /projects/{id}/tracer.gif` (needs the `image` extra).
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
  The UI has a format picker next to Export.
- ✅ **CCO growth variant** — `growth.cco_grow` (constrained constructive
  optimisation): terminals added one at a time by nearest-segment splitting,
  spaced ≥ `d_min`, Murray radii. Selectable via `POST /grow` `method=cco` and a
  method picker in the UI, beside space colonisation.
- ✅ **Multiple inlets / outlets** — the solver prescribes `p_in`/`p_out` at a
  *set* of boundary nodes (`Network.inlet_nodes`), not just the root; node
  roles are classified by **degree** so any leaf can be an inlet/terminal.
  `PUT/GET /projects/{id}/io` stores inlet/outlet points (snapped to vessel ends
  on run); the UI's **Inlets/Outlets** mode marks them (▲ inlet / ▼ outlet).

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

- ✅ **Task queue (optional RQ)** — job execution moved to `api.runner` +
  `run_job` (reconstructs the store from its root), so it runs in either the
  default thread pool **or** an RQ/Redis worker. `RQJobManager` behind the same
  `submit`/`get` seam; selected with `PERFUSION_QUEUE=rq` (+ `rq` extra, a running
  Redis, and `rq worker`), with graceful fallback to threads. Verified in-process
  with fakeredis.

**Next**

1. Server-side **mp4** (needs ffmpeg); couple the **3-D** grid to the network
   solver + vtk.js rendering; the optional React/Vite front-end rebuild.
2. Research tooling: parameter sweeps, inverse estimation, batch experiments.

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
| ✅ Heterogeneous permeability | per-cell `K(x)` field + harmonic-mean transmissibilities (`Grid.permeability` scalar-or-field, `rasterize_permeability`, MCP `heterogeneous_perfusion_lesion`) + an interactive low-K-lesion painting UI in Tissue mode; *todo:* anisotropic (tensor) K | `core/darcy` + `core/geometry` + web | done |
| Structural adaptation / autoregulation (Pries–Secomb) | radii evolve toward a wall-shear/metabolic set-point → emergent networks | iterate `growth` + solve | L |
| Poroelasticity (Biot) | tissue deformation ↔ perfusion coupling | new module | XL |

## 9.2 Numerics & solvers

- **Newton with analytic Jacobian** for the nonlinear solve (vs. the current fixed-point) — M.
- **AMG preconditioning** (pyamg) for the Darcy TPFA system; scales the 3-D solve — M.
- **MPFA-O flux** for anisotropic / non-K-orthogonal media (TPFA is inconsistent there) — M.
- **GPU / matrix-free** (cupy) for `darcy3d` on large grids — L.
- **Uncertainty quantification** (Monte-Carlo / polynomial chaos over parameters) — M.

## 9.3 Growth & geometry

- 3-D CCO and staged/optimised CCO (bifurcation-angle & Murray-exponent optimisation).
- Loopy venous networks / anastomoses (growth is tree-only today; veins are cyclic).
- Data-constrained growth (match a measured perfusion-density map or Strahler-order stats).
- Domain-aware growth (respect organ sub-regions / obstacles).

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
  `n_frames`), while relative rCBF/rCBV/rMTT are robust — the clinical norm. *(Next:
  block-circulant oSVD for delay-insensitivity; per-voxel model fitting.)*
- **Transit-time distributions & CTH** → MTT, CTH and oxygen-extraction limits
  (Jespersen–Østergaard). Reuses `core/timeofflight`. *(Started — see 9.6.)*

## 9.5 Optimisation, inverse problems, platform

- ✅ **Optimisation driver** over `optimize.objective` (Paper D flux-vs-TOF) —
  `optimize.optimize_tree` (Nelder-Mead / differential-evolution over the tree's
  branching asymmetries), exposed as the MCP `optimize_tree_config` tool.
- Calibration / parameter estimation to data; inverse recovery of boundary pressures or
  permeability from a perfusion map; adjoint gradients.
- **Frontend automated tests** (Playwright) — codify the CDP smoke flow for CI.
- Property-based conservation tests (hypothesis); run-comparison view; parameter-sweep UI;
  units safety (pint); network VTK `.vtp` export; benchmark harness (asv).

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
  `heterogeneous_perfusion_lesion` reports the in-vs-outside flux drop. *(Follow-up:
  anisotropic tensor K.)*
- ✅ **Oxygen / solute transport + extraction** (`core/oxygen.py`) — steady
  advection–reaction(–diffusion) of a blood-borne solute: arterial delivery, first-order
  tissue uptake at dimensionless Damköhler rate k, venous washout; a non-singular
  M-matrix solve (C ≥ 0, mass-conservative). Global OEF sits below the well-mixed Kety
  limit 1−e^{−k} exactly as transit heterogeneity predicts. `GET /projects/{id}/oxygen`
  (saturation / uptake maps + OEF), an Oxygen result card, MCP `oxygen_extraction_map`.
