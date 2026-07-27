# Developing perfusion

A practical map for extending the platform — for humans and AI assistants alike.
Read this with `CLAUDE.md` (conventions & invariants) and `docs/ROADMAP.md`
(where things are headed).

## Layers (depend only downward)

```
frontend/ (zero-build 2-D web UI)   ─┐
perfusion.api  (FastAPI + SQLite)    │  service / app layers
perfusion.mcp_server (MCP tools)     │  (optional extras: api, mcp, image)
io · viz · segmentation · growth · optimize
────────────────────────────────────┘
perfusion.core   (pure NumPy/SciPy engine — no I/O, viz, web, or optional deps)
```

**Rule:** physics and anything importable with only NumPy/SciPy goes in
`core`; anything with optional deps or side effects (files, network, plotting)
stays out of it. `core` must stay import-clean without any extra installed.

## Where does my change go?

| I want to… | Touch | Pattern to follow | Tests |
|---|---|---|---|
| add/adjust physics | `core/*` | vectorise via `Network.incidence()`; cite the thesis eq. | `tests/test_<module>.py` + a conservation/analytic check |
| add a network builder | `core/geometry.py` | return a `Network`; classify nodes by **degree** | round-trip / topology test |
| a new grid field or op | `core/darcy.py`, `core/timeofflight.py` | honour `Grid.mask`; column-major `order="F"` | linear-ramp / mask test |
| a result exporter | `io.py` | `fields_to_*` take arrays; `save_results_*` wrap a `HybridResult` | write + reparse |
| an API endpoint | `api/routes_{projects,geometry,solve}.py` (+ `schemas.py`, `store.py`) | add to the matching router; get store/jobs/project via `api.deps` (`Depends`); shared compute → `api/helpers.py`; long work → a job | `tests/test_api.py` (TestClient) |
| a UI capability | `frontend/` | state/geometry/api in `core.js`, drawing in `render.js`, vessel editing in `edit.js`, solve/results/tracer in `run.js`, viridis in `palette.js`, feature logic + wiring in `app.js`, markup in `index.html`; mode tabs; auto-save on run | `node --check`; live screenshot |
| an MCP tool | `mcp_server.py` | put logic in a plain function; wrap with `@server.tool()` | `tests/test_mcp.py` |
| a vessel-growth method | `growth.py` | produce `(pos, parent)` then `_assemble`; Murray radii | valid-tree + mass-conservation |

## Conventions that bite (see `CLAUDE.md` for the full list)

- **Column-major grids** (`order="F"`); `Grid.sub2ind(ix, iy) = ix + iy*nx`.
- **Node classification is by degree, not role**: `terminal_nodes` = degree-1
  minus `inlet_nodes`; boundary/inlet nodes carry the Dirichlet `p_in`/`p_out`.
- **Mass must balance**: `sum(terminal_flux) == 0`; arterial inflow == venous
  outflow. Most solver tests assert this — keep them green.
- **Preserve thesis equation numbers** in docstrings.

## Testing

```bash
uv run pytest                              # core + always-available tests
uv run --extra api --extra image --extra mcp pytest   # the FULL suite
uv run --extra api --extra image --extra mcp ruff check src tests scripts
```

API/image/MCP tests `pytest.importorskip` their extra, so a bare `uv run pytest`
**silently skips them** — use the full command above before committing anything
touching `api`, `segmentation`, or `mcp`.

## Running the pieces

```bash
uv run perfusion-api                       # web app + API at :8010  (extra: api)
uv run --extra mcp perfusion-mcp           # MCP server over stdio   (extra: mcp)
uv run perfusion-tree | perfusion-frog     # example drivers
```

**Jobs** run in an in-process thread pool by default (zero infra). For real
multi-worker hosting, set `PERFUSION_QUEUE=rq` (+ `--extra rq`, a running Redis at
`$REDIS_URL`, and one or more `uv run --extra rq rq worker perfusion` processes);
the API falls back to threads if Redis is unreachable. Job execution lives in
`api/runner.py` and `api/jobs.py:run_job` so a worker in another process can run
it — keep it importable and its args picklable.

## MCP server

`mcp_server.py` exposes the engine as MCP tools (`run_simulation`,
`describe_model`) so any MCP client — Claude Desktop, Claude Code, an agent — can
drive the model. The repo's `.mcp.json` registers it for Claude Code
automatically (`uv run --extra mcp perfusion-mcp`). To add a tool: write the
logic as a plain, testable function, then add a thin `@server.tool()` wrapper in
`build_server()` and a case to `tests/test_mcp.py`.

**Convention: new capabilities should get an MCP tool.** When you add a
user-facing feature (a new solve/analysis mode, growth method, exporter, …),
expose it here too so any AI client can drive it — keep the logic in a plain,
testable function and wrap it with `@server.tool()`. "Is it in MCP?" is part of
*done* for a new feature.

## Keeping the repo publishable

This is open research code. Keep it free of anything proprietary or personal:
no client/employer code, credentials, internal identifiers, or non-synthetic
data. Tests use synthetic or thesis-derived data only (`data/`, generated
references). When in doubt, don't commit it.
