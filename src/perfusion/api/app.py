"""FastAPI application wiring the compute core into a hosted service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .. import growth
from ..core.geometry import arterial_tree, network_from_polylines, rasterize_mask
from ..core.pipeline import run_hybrid
from .jobs import JobManager
from .schemas import (
    FieldResponse,
    GrowInfo,
    GrowRequest,
    JobInfo,
    MaskIn,
    ProjectCreate,
    ProjectInfo,
    RunRequest,
    VesselsIn,
)
from .store import Project, ProjectStore

_FIELDS = {"pressure": "pressure", "flux": "flux", "tof": "tof"}


def _polylines_to_network(polylines, root_xy=None):
    lines = [p.points for p in polylines]
    radii = [p.radius for p in polylines]
    return network_from_polylines(lines, radii=radii, root_xy=root_xy)


def _run_project(store: ProjectStore, project: Project, req: RunRequest) -> None:
    """Execute the hybrid simulation for a project and persist the result."""
    prm = project.params.to_core()
    domain_limit = project.grid.domain_limit
    mask = (rasterize_mask(project.grid.nx, project.grid.ny, domain_limit, project.regions)
            if project.regions else None)
    grid = project.grid.to_core(mask=mask)

    art = ven = None
    if req.use_vessels and project.has_vessels:
        art = _polylines_to_network(project.arteries)
        ven = _polylines_to_network(project.veins)
    if project.artery_network is not None:   # a grown tree takes precedence
        art = project.artery_network

    result = run_hybrid(prm, grid, domain_limit=domain_limit,
                        nonlinear=req.nonlinear, art=art, ven=ven)
    store.save_result(project, result)


def _info(project: Project) -> ProjectInfo:
    return ProjectInfo(id=project.id, name=project.name,
                       has_vessels=project.has_vessels,
                       has_mask=bool(project.regions), has_result=project.has_result)


def create_app(store: ProjectStore | None = None,
               jobs: JobManager | None = None) -> FastAPI:
    """Build the FastAPI app; inject a store/job-manager for testing."""
    from .. import __version__

    store = store or ProjectStore()
    jobs = jobs or JobManager()
    app = FastAPI(title="perfusion API", version=__version__,
                  summary="Multiscale blood-flow & perfusion modelling service")

    def _project(pid: str) -> Project:
        try:
            return store.get(pid)
        except KeyError:
            raise HTTPException(404, f"project {pid!r} not found") from None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/projects", response_model=ProjectInfo)
    def create_project(body: ProjectCreate) -> ProjectInfo:
        return _info(store.create(body.name, body.params, body.grid))

    @app.get("/projects", response_model=list[ProjectInfo])
    def list_projects() -> list[ProjectInfo]:
        return [_info(p) for p in store.list()]

    @app.get("/projects/{pid}", response_model=ProjectInfo)
    def get_project(pid: str) -> ProjectInfo:
        return _info(_project(pid))

    @app.put("/projects/{pid}/vessels", response_model=ProjectInfo)
    def set_vessels(pid: str, body: VesselsIn) -> ProjectInfo:
        project = _project(pid)
        project.arteries, project.veins = body.arteries, body.veins
        store.save(project)
        return _info(project)

    @app.put("/projects/{pid}/mask", response_model=ProjectInfo)
    def set_mask(pid: str, body: MaskIn) -> ProjectInfo:
        project = _project(pid)
        project.regions = [[list(pt) for pt in poly] for poly in body.regions]
        store.save(project)
        return _info(project)

    @app.get("/projects/{pid}/mask")
    def get_mask(pid: str) -> dict:
        return {"regions": _project(pid).regions}

    @app.post("/projects/{pid}/grow", response_model=GrowInfo)
    def grow_vessels(pid: str, body: GrowRequest | None = None) -> GrowInfo:
        """Grow the arterial tree into the tissue and store it for the next run."""
        project = _project(pid)
        req = body or GrowRequest()
        domain_limit = project.grid.domain_limit
        mask = (rasterize_mask(project.grid.nx, project.grid.ny, domain_limit, project.regions)
                if project.regions else np.ones((project.grid.nx, project.grid.ny), bool))

        base = (_polylines_to_network(project.arteries)
                if project.arteries else arterial_tree(project.params.to_core()))
        seed_nodes = None
        if req.from_segments:
            seed_nodes = [int(base.seg_end[s]) for s in req.from_segments
                          if 0 <= s < base.n_segments]

        network = growth.grow_in_mask(
            base, mask, domain_limit,
            n_points=req.n_points, seed=req.seed, seed_nodes=seed_nodes,
            step=req.step, influence_radius=req.influence_radius,
            kill_radius=req.kill_radius, leaf_radius=req.leaf_radius)
        project.artery_network = network
        store.clear_result(project)   # persist the tree; invalidate stale result
        return GrowInfo(id=project.id, n_segments=network.n_segments,
                        n_terminals=int(network.terminal_nodes.size))

    @app.post("/projects/{pid}/run", response_model=JobInfo)
    def run_project(pid: str, body: RunRequest | None = None) -> JobInfo:
        project = _project(pid)
        req = body or RunRequest()
        jid = jobs.submit(lambda: _run_project(store, project, req), project.id)
        state = jobs.get(jid)
        return JobInfo(id=jid, state=state["state"], error=state["error"],
                       project_id=project.id)

    @app.get("/jobs/{jid}", response_model=JobInfo)
    def job_status(jid: str) -> JobInfo:
        try:
            state = jobs.get(jid)
        except KeyError:
            raise HTTPException(404, f"job {jid!r} not found") from None
        return JobInfo(id=jid, **state)

    @app.get("/projects/{pid}/results")
    def results(pid: str) -> dict:
        project = _project(pid)
        if project.summary is None:
            raise HTTPException(409, "no result yet; run the simulation first")
        return project.summary

    @app.get("/projects/{pid}/results/field/{name}", response_model=FieldResponse)
    def result_field(pid: str, name: str) -> FieldResponse:
        project = _project(pid)
        if not project.has_result:
            raise HTTPException(409, "no result yet; run the simulation first")
        if name not in _FIELDS:
            raise HTTPException(404, f"unknown field {name!r}; use {list(_FIELDS)}")
        field = store.result_field(pid, _FIELDS[name])
        mask = store.result_field(pid, "mask")
        field = np.where(mask, field, np.nan)             # inactive cells -> null
        data = np.where(np.isfinite(field), field, None)  # NaN -> null (valid JSON)
        return FieldResponse(name=name, shape=field.shape, data=data.tolist())

    @app.get("/projects/{pid}/export")
    def export(pid: str):
        from fastapi.responses import FileResponse

        project = _project(pid)
        if not project.has_result:
            raise HTTPException(409, "no result yet; run the simulation first")
        return FileResponse(store.result_path(pid), media_type="application/octet-stream",
                            filename=f"{project.name or project.id}.npz")

    @app.get("/projects/{pid}/network")
    def network_geometry(pid: str) -> dict:
        """Segment geometry (domain frame) for overlaying grown/drawn vessels."""
        project = _project(pid)
        segments: list[dict] = []

        def _add(net, kind: str) -> None:
            for s in range(net.n_segments):
                a, b = int(net.seg_start[s]), int(net.seg_end[s])
                segments.append({
                    "x0": float(net.node_x[a]), "y0": float(net.node_y[a]),
                    "x1": float(net.node_x[b]), "y1": float(net.node_y[b]),
                    "r": float(net.seg_radius[s]), "kind": kind,
                })

        if project.artery_network is not None:
            _add(project.artery_network, "artery")
        elif project.arteries:
            _add(_polylines_to_network(project.arteries), "artery")
        if project.veins:
            _add(_polylines_to_network(project.veins), "vein")
        return {"segments": segments}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the static 2-D workspace at ``/`` if it is present (dev/editable)."""
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    if (frontend / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        # Mounted last, so the API routes above take precedence over "/".
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")


app = create_app()


def main() -> None:
    """Run the dev server (``perfusion-api``)."""
    import uvicorn

    from .store import _default_root
    print(f"perfusion-api: persisting projects under {_default_root()}  "
          f"(set PERFUSION_DATA to change)")
    uvicorn.run("perfusion.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
