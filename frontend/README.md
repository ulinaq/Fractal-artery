# frontend/ — 2-D web workspace

The browser front-end for the `perfusion` platform. This is a **zero-build**
single-page app (plain HTML/CSS/ES-modules) served directly by the
`perfusion.api` FastAPI back-end — no Node, npm, or bundler required.

## Run it

```bash
uv sync --extra api
uv run perfusion-api        # then open http://127.0.0.1:8000/
```

The API mounts this directory at `/` (see `_mount_frontend` in
`src/perfusion/api/app.py`), so the app and the API share one origin — no CORS,
no separate dev server.

## What it does

- **Project** — create a project (name, parameters, grid).
- **Draw vessels** — sketch arteries/veins on the canvas; each stroke is one
  segment, shared endpoints snap into branches (`PUT /projects/{id}/vessels`).
- **Draw tissue** — outline the organ in Tissue mode; the polygons become
  `Grid.mask`, so the Darcy bed and growth are organ-shaped
  (`PUT /projects/{id}/mask`).
- **Grow** — grow the arterial tree into the tissue by space colonisation
  (`POST /projects/{id}/grow`); the grown tree is overlaid on the result.
- **Run** — solve the hybrid model as an async job (`POST …/run` + poll
  `/jobs/{id}`).
- **Result** — view the pressure / flux / time-of-flight field as a
  colour-blind-safe *viridis* heatmap with the vascular network overlaid and a
  colour bar; read the inflow/outflow summary; **export** the raw `.npz`.
- **Share** — `?project=<id>` opens an existing project (and its result) by id.

## Files

| File | Role |
|---|---|
| `index.html` | layout & controls |
| `style.css`  | styling (no framework) |
| `app.js`     | API client, canvas drawing, field rendering, job polling |

## Roadmap

This satisfies Phase 2 (usable 2-D app) with the smallest possible toolchain.
The heavier React/TypeScript/Vite build named in `docs/ROADMAP.md` remains an
option for later — it would target the same OpenAPI contract, so the back-end
does not change. Planned next: hand-drawn tissue mask, flow-video playback, and
(Phase 4) 3-D views via vtk.js.
