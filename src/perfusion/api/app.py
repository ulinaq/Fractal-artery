"""FastAPI application wiring the compute core into a hosted service."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from .. import io
from ..core.geometry import network_from_polylines
from ..core.pipeline import run_hybrid
from .jobs import JobManager
from .schemas import (
    FieldResponse,
    JobInfo,
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


def _run_project(project: Project, req: RunRequest) -> None:
    """Execute the hybrid simulation for a project and store the result."""
    prm = project.params.to_core()
    grid = project.grid.to_core()
    domain_limit = project.grid.domain_limit

    art = ven = None
    if req.use_vessels and project.has_vessels:
        art = _polylines_to_network(project.arteries)
        ven = _polylines_to_network(project.veins)

    result = run_hybrid(prm, grid, domain_limit=domain_limit,
                        nonlinear=req.nonlinear, art=art, ven=ven)
    project.result = result
    project.summary = io.results_summary(result)


def _info(project: Project) -> ProjectInfo:
    return ProjectInfo(id=project.id, name=project.name,
                       has_vessels=project.has_vessels, has_result=project.has_result)


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
        return _info(project)

    @app.post("/projects/{pid}/run", response_model=JobInfo)
    def run_project(pid: str, body: RunRequest | None = None) -> JobInfo:
        project = _project(pid)
        req = body or RunRequest()
        jid = jobs.submit(lambda: _run_project(project, req), project.id)
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
        if project.result is None:
            raise HTTPException(409, "no result yet; run the simulation first")
        if name not in _FIELDS:
            raise HTTPException(404, f"unknown field {name!r}; use {list(_FIELDS)}")
        field = getattr(project.result, _FIELDS[name])
        data = np.where(np.isfinite(field), field, None)  # NaN -> null (valid JSON)
        return FieldResponse(name=name, shape=field.shape, data=data.tolist())

    @app.get("/projects/{pid}/export")
    def export(pid: str):
        from fastapi.responses import FileResponse

        project = _project(pid)
        if project.result is None:
            raise HTTPException(409, "no result yet; run the simulation first")
        path = Path(tempfile.gettempdir()) / f"perfusion_{project.id}.npz"
        io.save_results(project.result, path)
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{project.name or project.id}.npz")

    return app


app = create_app()


def main() -> None:
    """Run the dev server (``perfusion-api``)."""
    import uvicorn

    uvicorn.run("perfusion.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
