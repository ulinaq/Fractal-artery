# perfusion

[![CI](https://github.com/ulinaq/Fractal-artery/actions/workflows/ci.yml/badge.svg)](https://github.com/ulinaq/Fractal-artery/actions/workflows/ci.yml)

A Python implementation of the multiscale blood-circulation and perfusion models
from Ulin Nuha Abdul Qohar's PhD thesis, *Modeling and simulation of blood
circulation and perfusion* (University of Bergen, 2023).

The framework couples a **1D vascular network** (arteries + veins, Poiseuille
flow) to a **microcirculation model** — either a **Darcy porous-medium**
capillary bed (the *hybrid model*) or a **direct-connection** network — and adds
tracer transport and tracer-kinetic model (TKM) analysis for DCE-MRI. It is a
clean, consolidated port of the original MATLAB research code (now under
[`legacy/`](legacy/)).

## What's implemented

| Thesis paper | Topic | Modules |
|---|---|---|
| **A** | Nonlinear hybrid model (network + dual Darcy), vessel elasticity & junction losses | `network`, `darcy`, `pipeline` |
| **B** | Tracer flow in the hybrid model (single-compartment) | `transport` |
| **C** | Direct-connection microcirculation + TKM validation (1CM/MS/2CXM/2CUM) | `direct`, `transport`, `tkm` |
| **D** | Vascular-structure optimisation (flux vs. time-of-flight objectives) | `geometry`, `optimize`, `timeofflight` |

Core numerics: Poiseuille resistance with the **Fåhræus–Lindqvist** apparent
viscosity, a **TPFA** Darcy discretisation, a compactly supported **mollifier**
for the 1D→3D point-source coupling, a **Schur-complement**-style hybrid solve,
and a first-order **upwind time-of-flight** solver.

Beyond the thesis: a full **3-D hybrid** (3-D trees + 3-D Darcy bed with pressure,
flux and time-of-flight), vessel **growth** (space colonisation, CCO — 2-D and 3-D),
**pruning**, truly volumetric 3-D fractal trees, a library of **organ tissue domains**
with anatomical vessel entry points, hematocrit phase separation, multi-compartment
beds, oxygen extraction, structural adaptation, Windkessel transients, UQ/Sobol and
calibration — all reachable from the web app, the API and MCP.

## Install & run (uv)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                      # core deps into .venv
uv sync --extra image        # + imageio/scikit-image (for image segmentation)
uv sync --extra api          # + fastapi/uvicorn/pydantic (for the web back-end)
uv sync --extra mcp          # + the MCP SDK (to expose the engine as MCP tools)
uv sync --extra video        # + imageio-ffmpeg (server-side MP4 of the flow video; bundles ffmpeg)

uv run pytest                # run the test suite
uv run --extra api --extra image --extra ui pytest tests/ui   # browser tests (Playwright)
uv run perfusion-tree        # Paper D: hybrid on a synthetic fractal tree
uv run perfusion-frog        # Paper A: hybrid on the frog-tongue domain
uv run python -m perfusion.examples.tkm_demo   # Paper C: TKM comparison
uv run python scripts/generate_data.py         # regenerate bundled datasets
uv run perfusion-api         # start the FastAPI back-end (http://127.0.0.1:8010)
uv run --extra mcp perfusion-mcp   # start the MCP server (stdio)
uv run perfusion-darcy3d     # 3-D Darcy source→sink slices
uv run perfusion-hybrid3d    # Phase 4: full 3-D hybrid (3-D trees + 3-D Darcy)
```

### Web app

`uv run perfusion-api` starts the FastAPI back-end **and** serves a zero-build web
app (no Node/npm) at `http://127.0.0.1:8010/`. Every canvas fills the window, stays
sharp on HiDPI screens and **zooms/pans** (scroll = zoom about the cursor,
right-drag / Space+drag = pan, ＋ − ⤢ buttons).

**2-D workspace** (`/`) — three tabs:

- **Vessels** — edit the artery/vein **networks** directly: click to start a vessel
  and keep clicking to extend it, click a node then empty space to branch, drag
  nodes, **click a segment to change its radius** at any time, double-click to insert
  a node, Shift+click to join, Del to delete. **Inlets/outlets are nodes** — select
  one and press *Set as inlet/outlet* (`I`). Also: segment a vessel image, **grow**
  (space colonisation / CCO) or **prune** (remove the outermost generations), Strahler
  morphometry and Pries–Secomb radius adaptation.
- **Tissue** — draw/edit the organ outline, low-K lesions and growth hotspots
  vertex by vertex (drag, insert, delete, move), or drop in an **organ preset**
  (heart, kidney, brain, liver, lung, …) and adjust it.
- **Result** — pressure / flux / time-of-flight heatmaps, a contrast-bolus **flow
  video** (+ GIF / MP4), transit-time distribution, tracer-kinetic fits and maps,
  pulsatile, multi-compartment, oxygen and uncertainty analyses, and export.

🐸 **Frog** loads a real segmented frog-tongue system in one click. Set a New
project's **domain size** and grid in Parameters. Projects are shareable by URL
(`/?project=<id>`), save/load as one `.json` file, have global undo/redo, and
**persist across restarts** (SQLite + a blob store under `$PERFUSION_DATA`).

**🌳 Tree explorer** (`/explorer.html`) — grow an artery + vein into a tissue in
**2-D or 3-D** with no project: pick an **organ** (each with its *anatomical* artery
and vein entry — renal hilum, aortic root, porta hepatis, circle of Willis, …),
move the roots, set the **generations** (each step adds a level of branches),
branching asymmetry / Murray exponent, 3-D **branch-plane twist** and randomness,
and a **space-filling** algorithm (colonisation steps or CCO terminals); scale the
**object size** (opens at ×4, up to ×16) and grid resolution; see the solved
pressure / flux / time-of-flight field — computed in 3-D for 3-D trees — plus
Strahler orders, inflow and a homogeneity objective, or *Animate growth*.
**Optimise trees** searches the branching parameters (γ, length : radius, Murray
exponent, vein calibre, 3-D twist) for the most homogeneous perfusion **in the chosen
organ** — its roots, fill and generations, 2-D or 3-D — re-scored at your resolution;
**📤 Use optimised trees in simulation** then opens them in the simulation page.
**📤 Open as project** turns the setup into a project: 2-D opens in the workspace
(outline + trees editable), 3-D opens in the 3-D viewer to run and inspect.

**🧊 3-D viewer** (`/viewer3d.html`) renders a 3-D hybrid result (trees + voxel
field) in a rotatable, dependency-free canvas. For a **3-D project** it is a full
workbench: run the 3-D solve; grow (CCO / colonisation), prune or Murray-adapt the 3-D
trees, or **edit them by hand** (select, drag, branch, split, set inlets/outlets, undo); compute the **🌊 flow** (a contrast bolus advected through the voxel bed, played
back in 3-D) with a tracer-kinetic fit; and map pressure / flux / time-of-flight,
O₂ saturation / consumption, the Fp / vp / CBV / MTT kinetic maps, or a
**multi-compartment** bed (capillary exchange, arteriole / venule pressure). Its
**Uncertainty & calibration** card runs Monte-Carlo / Sobol UQ over p_in, μ and K and
fits one input to a measured inflow / transit time — all on the 3-D model. Switch the
model to **direct — terminal connections** to close the 3-D trees with resistive links
instead of a Darcy bed: the viewer draws the links and vessels weighted by flow, and 🌊
flow times the bolus along the network. Interactive API docs are at `/docs`.
See [`frontend/`](frontend/README.md) for the UI in detail.

### Back-end API

The same service exposes the model as a hosted API — create a project, run a
simulation as an async job, and fetch/export results, all under one origin as
the UI above.

```bash
uv run perfusion-api
curl -s localhost:8010/health
# create a project, run it, poll the job, get the summary:
pid=$(curl -s -XPOST localhost:8010/projects -H 'content-type: application/json' \
      -d '{"name":"demo"}' | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')
jid=$(curl -s -XPOST localhost:8010/projects/$pid/run -d '{}' \
      -H 'content-type: application/json' | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')
curl -s localhost:8010/jobs/$jid          # -> {"state":"done",...}
curl -s localhost:8010/projects/$pid/results
```

Main endpoints (full OpenAPI at `/docs`):

| Area | Endpoints |
|---|---|
| Projects | `projects` (CRUD), `projects/frog`, `projects/import`, `projects/{id}/document` (portable `.json`) |
| Geometry | `projects/{id}/editable` (GET/PUT the artery/vein networks — create, edit, remove; `inlet_nodes` = inlets/outlets), `…/image?kind=artery\|vein` (segment), `…/grow`, `…/prune`, `…/adapt`, `…/morphometry`, `…/mask`, `…/lesions`, `…/hotspots`, `…/network`, legacy `…/vessels` + `…/io` |
| Solve & results | `…/run` (job), `jobs/{id}`, `…/results[/field/{name}]`, `…/tracer` (+ `tracer.gif`, `tracer.mp4`), `…/transit`, `…/kinetics[/maps]`, `…/oxygen`, `…/multicompartment`, `…/transient`, `…/uq`, `…/calibrate`, `…/export?fmt=npz\|csv\|vtk\|vtp`, `…/bundle`, `…/view3d` |
| Explorer | `explore/grow` (roots + organ + generations + fill, 2-D/3-D, solve), `explore`, `explore/optimize`, `shapes`, `shapes/polygon` |

### MCP server

`uv run --extra mcp perfusion-mcp` starts a [Model Context
Protocol](https://modelcontextprotocol.io) server (stdio) exposing the engine as
tools, so any MCP client (Claude Desktop, Claude Code, an agent) can drive it:
`run_simulation`, `run_sweep`, `describe_model`, `grow_trees_in_tissue` (organ,
2-D/3-D, generations, fill), `optimize_trees_in_tissue` (organ-aware Paper-D
optimisation), `tissue_perfusion_analysis` (2-D/3-D grow → solve → adapt / transit /
O₂ / tracer fit / multi-compartment), `tissue_uncertainty_calibration` (UQ, Sobol and
calibration on an organ's 2-D/3-D model), `prune_vessel_tree`, `data_constrained_tree_growth`,
`optimize_tree_config`, `adapt_vessel_network`, `strahler_morphometry`,
`fit_tracer_kinetics`, `perfusion_parameter_maps`, `direct_transit_metrics`,
`direct_model_in_tissue` (the direct model in a 2-D/3-D organ),
`pulsatile_windkessel`, `multicompartment_perfusion`, `oxygen_extraction_map`,
`heterogeneous_perfusion_lesion`, `anisotropic_perfusion_bed`,
`calibrate_parameter_to_flow`, `uncertainty_quantification`,
`sobol_sensitivity_indices`. The bundled [`.mcp.json`](.mcp.json) registers it for
Claude Code automatically when working in this repo.

Example figures are written to `outputs/`.

## Bundled data

- `data/frog_domain_low.png`, `data/frog_trees.mat` — the segmented frog-tongue
  outline and vascular networks (loaders in `perfusion.io`).
- `data/synthetic_perfusion.npz` — a small reusable DCE-MRI dataset (time grid,
  AIF, tissue curves for known `(Fp, vp)`); load with
  `perfusion.io.load_synthetic_perfusion()`.
- `tests/data/reference_hybrid.npz` — a locked reference solution used to catch
  numerical regressions.

Both `.npz` files are regenerated by `scripts/generate_data.py`.

## Quick start

```python
import numpy as np
from perfusion import Parameters, Grid, run_hybrid

prm = Parameters(p_in=10.0, p_out=0.0)     # blood/flow + fractal-tree geometry
grid = Grid(nx=140, ny=160, lx=140, ly=160) # capillary bed
res = run_hybrid(prm, grid)                 # build trees, couple to Darcy, solve

res.pressure   # (nx, ny) capillary pressure field
res.flux       # (nx, ny) Darcy flux magnitude
res.tof        # (nx, ny) time-of-flight
res.solution.q_a[0]   # arterial trunk inflow
```

Segment the real frog vasculature from images and solve the full organ system:

```python
from perfusion import Parameters, run_hybrid
from perfusion import segmentation as seg

system = seg.load_frog_system()          # segments artery + vein images + mask
res = run_hybrid(Parameters(), system.grid, domain_limit=system.domain_limit,
                 art=system.artery, ven=system.vein)   # couples them via Darcy
# segment a single vessel image directly:
net = seg.segment_network(image, root_xy=(x0, y0))     # -> a rooted Network
```

Grow a vascular tree into a tissue region (space colonisation) and solve it:

```python
import numpy as np
from perfusion import Grid, Parameters, growth, run_hybrid
from perfusion.core.geometry import network_from_polylines

trunk = network_from_polylines([[(50, 2), (50, 40)]], radii=2.0, root_xy=(50, 2))
grid = Grid(nx=40, ny=50, lx=100, ly=100)
tree = growth.grow_in_mask(trunk, grid.mask, (0, 100, 0, 100), n_points=300)
res = run_hybrid(Parameters(), grid, domain_limit=(0, 100, 0, 100), art=tree)
```

Direct-connection model + tracer-kinetic fitting:

```python
from perfusion import arterial_tree, venous_tree, direct, tkm
import numpy as np

art, ven = arterial_tree(prm), venous_tree(prm)
conns = direct.connect_terminals(art, ven, rho_c=60.0)
flow = direct.solve_direct(art, ven, conns, p_in=75.0, p_out=20.0)

t = np.arange(0, 120, 0.5)
ca = tkm.gamma_variate_aif(t, t0=5.0)
C = tkm.one_compartment(t, ca, Fp=40.0, vp=8.0)   # synthetic tissue curve
fit = tkm.fit(t, ca, C, model="2CXM")             # recover Fp, vp, PS, ve
```

## Documentation

- **[`docs/MODEL_REFERENCE.pdf`](docs/MODEL_REFERENCE.pdf)** — a standalone
  reference explaining the concepts, equations and numerical methods, with a
  concept → equation → code map. Source: `docs/MODEL_REFERENCE.md` (rebuild the
  PDF with the pandoc → Chrome recipe in `CLAUDE.md`).
- **[`docs/MODELS.md`](docs/MODELS.md)** — every module's governing equation
  (Poiseuille/Darcy/TPFA, coupling, transport, TKM, oxygen, adaptation, growth,
  3-D post-processing, organ domains) with thesis equation numbers.
- **[`docs/DEVELOPING.md`](docs/DEVELOPING.md)** — how to extend the platform: a
  feature → files → pattern → tests map, conventions, and how to run the pieces.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — the development plan: target
  architecture (FastAPI back-end + web front-end, hosted), feature roadmap
  (2-D → 3-D, vessel growth, larger systems), and the source layout.
- **`CLAUDE.md`** — implementation notes, invariants, and conventions for
  contributors.

## Repository layout

```
src/perfusion/          # the package
  core/                 # pure compute engine (no I/O, viz, or web deps)
    parameters.py       #   model parameters + Murray bifurcation geometry
    viscosity.py        #   Fåhræus–Lindqvist apparent viscosity
    geometry.py         #   Network dataclass + fractal-tree generation
    darcy.py            #   TPFA Darcy bed, mollifier, terminal response, fields
    network.py          #   hybrid solver (linear + nonlinear w/ junction losses)
    direct.py           #   direct-connection microcirculation (Paper C)
    transport.py        #   passive tracer advection (theta rule)
    tkm.py              #   tracer-kinetic models + fitting
    timeofflight.py     #   flux magnitude + upwind time-of-flight (2-D and 3-D)
    pipeline.py         #   one-call end-to-end hybrid run
    darcy3d.py          #   3-D Darcy grid/TPFA, face fluxes, upwind operator
    pipeline3d.py       #   3-D hybrid: 3-D trees + 3-D Darcy → pressure, flux, TOF
    …                   #   hematocrit, multicompartment, oxygen, adaptation, morphometry, …
  segmentation.py       # image → vascular Network (skeletonise + graph trace)
  growth.py             # grow vessels into tissue: space colonisation + CCO (2-D/3-D)
  organs.py             # organ tissue shapes (2-D outlines, 3-D masks) + anatomical roots
  treegen.py            # tree-explorer service: roots → fractal skeleton → fill → solve
  optimize.py           # vascular-structure optimisation objectives (Paper D)
  uq.py, inverse.py     # uncertainty / Sobol sensitivity, calibration
  io.py, viz.py         # .mat/PNG loaders + result exporters, matplotlib helpers
  api/                  # FastAPI back-end: projects, jobs, results (extra: api)
  mcp_server.py         # MCP server exposing the engine as tools (extra: mcp)
  examples/             # runnable drivers for Papers A/C/D
tests/                  # pytest suite (physics sanity checks + API)
data/                   # example assets (frog domain, segmented trees)
frontend/               # zero-build web app (workspace, tree explorer, 3-D viewer), served at /
legacy/                 # original MATLAB source (reference only)
```

The public API is stable — `from perfusion import Grid, Parameters, run_hybrid,
arterial_tree, venous_tree, Network` and the module handles (`perfusion.darcy`,
`perfusion.geometry`, …) work; deep imports live under `perfusion.core.*`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for delivery status and what is left.

## Notes & scope

- This is an **idiomatic, consolidated** reimplementation, not a line-by-line
  translation. The many duplicated MATLAB experiment folders
  (`Optim/`, `Optim2/`, `OptimPaper/`, `Visit_UK/`, …) are collapsed into single
  parametrised modules. The originals remain in `legacy/` for reference.
- The hybrid solver uses **Dirichlet–Dirichlet** boundary conditions: `p_in` is
  prescribed at the arterial inlet node(s) and `p_out` at the venous outlet
  node(s) — any number of each (`Network.inlet_nodes`, default the root); the
  Darcy datum is fixed by pinning one cell, and terminal fluxes are rebalanced so
  the pin carries no net flux (as in the thesis).
- The frog example (`perfusion-frog`, and 🐸 in the web app) runs on the **real
  segmented** frog-tongue artery/vein networks and tongue mask
  (`segmentation.load_frog_system`).

## License

See [`license.txt`](license.txt). Please cite the thesis and the associated
papers (Royal Society Open Science, ENUMATH 2019, Magnetic Resonance in
Medicine) if you use this code.
