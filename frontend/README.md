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
- **Project** — create a project (name, parameters, grid).
- **Draw vessels** — sketch arteries/veins on the canvas; each stroke is one
  segment, shared endpoints snap into branches (`PUT /projects/{id}/vessels`).
- **Segment an image** — upload a vessel image; it is skeletonised into the
  artery or vein network (per the artery/vein selector) placed in the domain
  (`POST /projects/{id}/image?kind=…`, `image` extra).
- **Draw tissue** — outline the organ in Tissue mode; the polygons become
  `Grid.mask`, so the Darcy bed and growth are organ-shaped
  (`PUT /projects/{id}/mask`).
- **Inlets / Outlets** — in I/O mode, click near a vessel end to mark an inlet
  (▲, artery, `p_in`) or outlet (▼, vein, `p_out`); any number of each
  (`PUT /projects/{id}/io`). Empty = the single root inlet/outlet.
- **Grow** — grow the arterial tree into the tissue (space colonisation or CCO,
  via the method picker; `POST /projects/{id}/grow`); the grown tree is overlaid.
- **Run** — solve the hybrid model as an async job (`POST …/run` + poll
  `/jobs/{id}`).
- **Result** — view the pressure / flux / time-of-flight field as a
  colour-blind-safe *viridis* heatmap with the vascular network overlaid and a
  colour bar; read the inflow/outflow summary; **export** the fields as `.npz`,
  CSV, or ParaView `.vti` (format picker next to Export), or grab a **PNG** of the
  current view (📷). Actions lock while a job runs; the project id is a shareable
  `?project=` link.
- **Flow video** — play/scrub an animation of a contrast bolus washing through
  the tissue (`POST/GET /projects/{id}/tracer`), and download it as an animated
  GIF (`tracer.gif`, needs the `image` extra).
- **Share** — `?project=<id>` opens an existing project (and its result) by id.

- **3-D viewer** (`viewer3d.html`) — a separate page (linked from the header)
  that renders results in a rotatable, **dependency-free canvas** projection (no
  WebGL/vtk.js, works offline). Two models via `GET /demo3d?model=`: a synthetic
  3-D hybrid (trees + voxel field) and the **frog tongue** as a 3-D pressure-relief
  surface with the vasculature at its base. Deep-linkable (`?model=frog`).

The left-panel **tools are contextual**: picking a tab above the canvas
(Vessels · Tissue · Inlets/Outlets · Result) shows just that tool card; Project,
Parameters and Run stay put — so there's far less to scroll.

## Files

| File | Role |
|---|---|
| `index.html` | layout & controls |
| `style.css`  | styling (no framework) |
| `app.js`     | API client, canvas drawing, field rendering, job polling |
| `viewer3d.html` / `viewer3d.js` | standalone rotatable 3-D result viewer |

## Roadmap

This satisfies Phase 2 (usable 2-D app) with the smallest possible toolchain.
The heavier React/TypeScript/Vite build named in `docs/ROADMAP.md` remains an
option for later — it would target the same OpenAPI contract, so the back-end
does not change. Planned next: hand-drawn tissue mask, flow-video playback, and
(Phase 4) 3-D views via vtk.js.
