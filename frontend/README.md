# frontend/ — 2-D web workspace

The browser front-end for the `perfusion` platform. This is a **zero-build**
single-page app (plain HTML/CSS/ES-modules) served directly by the
`perfusion.api` FastAPI back-end — no Node, npm, or bundler required.

## Run it

```bash
uv sync --extra api
uv run perfusion-api        # then open http://127.0.0.1:8010/
```

The API mounts this directory at `/` (see `_mount_frontend` in
`src/perfusion/api/app.py`), so the app and the API share one origin — no CORS,
no separate dev server.

## What it does

- **Frog example** — one click (`POST /projects/frog`, `image` extra) loads the
  bundled segmented frog-tongue system — arterial + venous networks placed in the
  domain and the tongue as the organ mask — so the whole pipeline is instantly
  testable on real data.
- **Project** — create a project (name, parameters, physical **domain size** and
  grid — ny follows the aspect), save/load it as a `.json` file, undo/redo.
- **Vessels tab — edit the network directly.** Vessels are a node/edge network
  (each edge has its own radius), edited on the canvas: click empty space to
  start a vessel and keep clicking to extend it; click a node, then empty space,
  to branch from it; drag a node to move it; **click a vessel to select that
  segment** and set its radius (slider or number) at any time; double-click a
  vessel (or *＋ Insert node*) to split it; ⇧ Shift+click a node/vessel to join
  to it; Del deletes the selected node or segment; Esc finishes. With nothing
  selected the radius control sets the radius of new segments.
  **Inlets/outlets are nodes**: select one and press *Set as inlet* (artery,
  `p_in`) / *Set as outlet* (vein, `p_out`) or `I` — stored as the network's
  `inlet_nodes`; a new vessel's first node is its feed by default. Save with
  *Save vessels* (`PUT /projects/{id}/editable`, which also creates/removes
  trees); Run/Grow/Adapt save pending edits automatically.
- **Segment an image** — upload a vessel image; it is skeletonised into the
  artery or vein network (per the artery/vein selector) placed in the domain
  (`POST /projects/{id}/image?kind=…`, `image` extra).
- **Zoom / pan everywhere** (`viewport.js`): canvases fill the window (HiDPI), scroll
  zooms about the cursor, right-drag / Space+drag pans (plain drag in Result), ＋ − ⤢.
- **Domain size + organ presets**: Parameters sets a New project's physical
  width/height (grid ny follows the aspect); the Tissue tab's *organ preset* drops
  an editable organ outline (`POST /shapes/polygon`).
- **Tissue tab — the same node editing for polygons.** Pick a layer (organ
  outline → `Grid.mask`, low-K lesion, growth hotspot); click to add points,
  click the first point / Enter to close; drag a point, click an outline to
  insert one, drag inside to move a region, Del deletes. Only the chosen layer
  is editable, so lesions/hotspots can be drawn inside the organ
  (`PUT /projects/{id}/mask|lesions|hotspots`).
- **Prune** — the inverse of Grow: remove the outermost `steps` generations of
  the artery or vein (each step cuts every free-end branch back to its junction;
  the trunk to the inlet/outlet is kept) — `POST /projects/{id}/prune`; undoable.
- **Grow** — grow the arterial tree into the tissue (space colonisation or CCO,
  via the method picker; `POST /projects/{id}/grow`); the grown tree is overlaid.
- **Run** — solve the hybrid model as an async job (`POST …/run` + poll
  `/jobs/{id}`). Options: nonlinear junction losses, phase separation, wall
  elasticity and **shear-thinning blood** (Carreau law, time constant λ — the
  summary then shows the viscosity-factor range); the 3-D viewer has the same
  shear-thinning option.
- **Result** — view the pressure / flux / time-of-flight field as a
  colour-blind-safe *viridis* heatmap with the vascular network overlaid and a
  colour bar; read the inflow/outflow summary; **export** the fields as `.npz`,
  CSV, or ParaView `.vti` (format picker next to Export), or grab a **PNG** of the
  current view (📷). Actions lock while a job runs; the project id is a shareable
  `?project=` link.
- **Loops & occlusion** — the Vessels tab's *Grow / prune / loops* card has ⟳ Add loops
  (collateral anastomoses between nearby branches of the chosen tree; terminals kept)
  and a selected vessel has ⛔ Occlude (radius × 0.001, an embolus); the 3-D viewer has
  ⟳ Loops and ⛔ Occlude too. Pair with Compare runs to see the collateral rescue.
- **Compare runs (A/B)** (`compare.js`, Result tab and 3-D viewer) — 📌 Keep the
  current result as a named baseline, change something and re-run, then Compare: a
  table of what changed (inflow, MTT, …, field means) and Δ pressure / flux / TOF maps
  on a diverging ramp (blue = lower than the baseline, red = higher). **Batch
  experiment**: pick an input (p_in, p_out, μ, K; tree depth in 2-D) and values, ▶ Run
  batch — every run is kept and listed in a sweep table; click a row to compare.
- **Fit to DCE-MRI** (`dcefit.js`, Result tab and the 3-D viewer's analyses card) —
  paste `t, tissue, aif` CSV (or take this project's own tracer curve), tick the inputs
  to fit with bounds, Fit: values ± 1σ, R², warnings for correlated / bound-limited
  inputs, and a measured-vs-model chart.
- **Flow video** — play/scrub an animation of a contrast bolus washing through
  the tissue (`POST/GET /projects/{id}/tracer`), and download it as an animated
  GIF (`tracer.gif`, needs the `image` extra) or MP4 (`tracer.mp4`, needs the
  `video` extra — `imageio-ffmpeg` bundles its own ffmpeg).
- **Share** — `?project=<id>` opens an existing project (and its result) by id.

- **Tree explorer** (`explorer.html`) — grow an artery + vein into a tissue, in
  **2-D or 3-D**, with no project needed (`POST /explore/grow`, logic in
  `perfusion.treegen`). Set the **initial condition** (drag the ▲/▼ roots and their
  aim handles in 2-D; x/y/z + trunk direction in 3-D — each organ starts with its
  *anatomical* artery/vein entry, ⟲ Organ default restores it), the **tissue shape**
  (organ presets from `perfusion.organs` — heart, kidney, brain, liver, lung,
  tongue, tumour, placenta, … — or drag/insert/delete outline points for a custom
  2-D shape), the **domain size** (explore a bigger object) and **grid resolution**,
  3-D branch-plane **twist** + **randomness**, the fractal skeleton's **generations** (−/＋
  adds a level of branches), and a **space-filling** algorithm — space
  colonisation (growth steps, attraction points, step length, influence radius) or
  CCO (terminals) — then see the solved field, Strahler orders and inflow.
  *Animate growth* replays it generation by generation, then step by step.
  **Optimise trees** (`POST /explore/optimize`) tunes the ticked branching parameters
  for the chosen homogeneity objective in the current organ setup (2-D or 3-D),
  searching on a coarser grid and re-scoring at your resolution (your settings are
  kept if nothing beats them); **📤 Use optimised trees in simulation** opens the
  result as a project — the 2-D workspace or the 3-D viewer.
  **📤 Open as project** saves the setup as a project (`POST /explore/project`):
  2-D → this workspace (editable), 3-D → the 3-D viewer (run, grow / prune / adapt,
  🌊 flow playback + kinetics, oxygen and kinetic-map fields).
- **Default trees** — with no vessels drawn, the fractal trees a run would use are
  shown dashed in the Vessels tab (fitted to the project's domain).
- **3-D viewer** (`viewer3d.html`) — a separate page (linked from the header)
  that renders results in a rotatable, **dependency-free canvas** projection (no
  WebGL/vtk.js, works offline). Two models via `GET /demo3d?model=`: a synthetic
  3-D hybrid (trees + voxel field) and the **frog tongue** as a 3-D pressure-relief
  surface with the vasculature at its base. Deep-linkable (`?model=frog`).
  Opened on a **3-D project** (`?project=<id>`) it is a workbench: run, grow / prune /
  adapt, 🌊 flow + kinetics, oxygen and multi-compartment fields, and an
  **Uncertainty & calibration** card (Pearson / Sobol UQ over p_in, μ, K; fit one input
  to a measured value — it warns when the fit stops at a search bound). **✏️ edit the
  trees by hand** (`edit3d.js`): click to select a node or vessel, drag a node (it moves
  in the screen plane — rotate to reach depth), Shift+click to branch, S split, I
  inlet/outlet, Del delete, ⌘/Ctrl+Z undo; each edit saves via `PUT /editable`. The
  **model** picker also runs the **direct — terminal connections** model (ρc search
  radius): links (green) and vessels are drawn by flow, and 🌊 flow plays the bolus
  timed along the network. An export row saves the fields (`.npz` / `.csv` / `.vti`), the
  vessels (`.vtp`), a zip bundle, or a 📷 PNG of the view; after 🌊 flow, ↓ GIF / ↓ MP4
  download the bolus rendered on the server (rotating maximum-intensity projection).
  **Lesions & hotspots** (`regions3d.js`): ＋ adds a sphere at the tissue centre; edit
  x/y/z, radius and K / weight in the card, or drag its centre dot in the view.

The left-panel **tools are contextual**: picking a tab above the canvas
(Vessels · Tissue · Result, keys 1–3) shows just that tool card; Project,
Parameters and Run stay put — so there's far less to scroll. Legacy projects
(hand-drawn polylines, inlet/outlet *points*) are converted on open: strokes
are vectorised into the network and points snap onto the nearest end nodes.

## Files

| File | Role |
|---|---|
| `index.html` | layout & controls |
| `style.css`  | styling (no framework) |
| `core.js`    | domain/canvas geometry, API client, shared state |
| `app.js`     | project lifecycle, history (undo/redo), tabs, keyboard, wiring |
| `edit.js`    | Vessels tab: node/edge network editor + inlet/outlet nodes |
| `tissue.js`  | Tissue tab: polygon (organ / lesion / hotspot) editor |
| `render.js`  | all canvas drawing |
| `viewport.js`| shared canvas sizing (fills the stage, HiDPI) + zoom/pan/fit for every view |
| `run.js`     | solve, results, tracer, analyses |
| `viewer3d.html` / `viewer3d.js` | standalone rotatable 3-D result viewer |
| `edit3d.js` | hand-editing the 3-D trees in the viewer (pick, drag, branch, undo) |
| `regions3d.js` | lesion / hotspot spheres on a 3-D project (card + drag in the viewer) |
| `compare.js` | run comparison (A/B) panel shared by the workspace and the 3-D viewer |
| `dcefit.js` | multi-parameter DCE-MRI fit panel (workspace + 3-D viewer) |
| `uqview.js` | UQ result tables (Pearson / Sobol bars), shared by workspace + viewer |
| `explorer.html` / `explorer.js` | tree explorer: roots, tissue shape, 2-D/3-D growth |

## Roadmap

The zero-build stack stays deliberately small. Still open for the UI (see
`docs/ROADMAP.md` §11): a run-comparison view, and an
optional React/Vite + vtk.js rebuild against the same API.
