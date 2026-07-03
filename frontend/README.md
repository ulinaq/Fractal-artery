# frontend/ — web UI (planned, Phase 2)

The browser front-end for the `perfusion` platform. **Not built yet** — this
directory is a placeholder that marks where it lives; see `docs/ROADMAP.md`.

## Plan

- **Stack**: React + TypeScript (Vite).
- **Talks to**: the `perfusion.api` FastAPI back-end (Phase 1) over its OpenAPI
  contract; WebSocket for job progress.
- **Workspace features**:
  - upload a blood-vessel image; preview/adjust segmentation;
  - **draw** the tissue domain (mask) and vessels (polylines) on a canvas;
  - configure parameters and **run** simulations (with progress);
  - **grow** vessels from selected segments;
  - view 2-D fields (pressure / flux / time-of-flight) and **flow videos**;
  - 3-D views via **vtk.js** (Phase 4);
  - **export** figures, videos and raw results.

## Build order

The API (Phase 1) defines the contract first, so this app can be developed
against a running back-end. Until then, the Python package and its `examples/`
provide the same computations programmatically.
