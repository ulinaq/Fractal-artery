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
- **Jobs**: simulations can take seconds-to-minutes. Run them as background jobs
  with progress over WebSocket. Start with in-process `BackgroundTasks`; move to a
  task queue (RQ/Celery + Redis) for real multi-user hosting.
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
| Grow vessels from selected segments | pick + run | `POST /projects/{id}/grow` | **new** `growth` (CCO / space colonisation) |
| Run simulation | run + progress | `POST /projects/{id}/run` (job) | `core.pipeline.run_hybrid` |
| 2-D result views | canvas/Plotly | `GET /results/{id}/field/{name}` | `core` fields → JSON/PNG |
| 3-D result views | vtk.js | `GET /results/{id}/mesh` | **new** VTK/glTF export (Phase 4) |
| Flow video (2-D/3-D) | ✅ player + scrubber | ✅ `POST/GET /projects/{id}/tracer`, `…/tracer.gif` | `core.transport` (bolus advection) + GIF |
| Export results | download | `GET /results/{id}/export?fmt=` | **new** `io` exporters (npz/CSV/VTK/JSON/MP4) |
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
- ✅ **Image upload + segmentation** — `POST /projects/{id}/image` segments an
  uploaded vessel image (`segmentation.segment_network`) and places it in the
  project's domain (`segmentation.network_in_domain`) as the arterial tree; the
  UI has a file picker and shows the segmented tree. Needs the `image` extra.

**Next**

1. **Task queue** — move jobs from the in-process thread pool to RQ/Celery +
   Redis for real multi-user hosting.
2. **Segment veins too** — a second uploaded image → the venous network (needs a
   `vein_network` alongside the arterial one).
3. **CCO growth variant**; **VTK/CSV export**; server-side **mp4** (needs
   ffmpeg); then the optional React/Vite rebuild and Phase 4 (3-D + vtk.js).

Each step is independently useful and testable; the API contract lets the
front-end be built in parallel.
