"""FastAPI application wiring the compute core into a hosted service."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from .. import growth
from .. import io as perfio
from ..core.geometry import arterial_tree, rasterize_mask
from . import runner
from .jobs import JobManager, RQJobManager
from .schemas import (
    FieldResponse,
    GrowInfo,
    GrowRequest,
    IOIn,
    JobInfo,
    MaskIn,
    ProjectCreate,
    ProjectInfo,
    RunRequest,
    TracerInfo,
    TracerRequest,
    VesselsIn,
)
from .store import Project, ProjectStore

_FIELDS = {"pressure": "pressure", "flux": "flux", "tof": "tof"}


def _default_jobs(store: ProjectStore):
    """Pick the job backend: RQ if ``PERFUSION_QUEUE=rq`` (and reachable), else threads."""
    import os

    if os.environ.get("PERFUSION_QUEUE", "").lower() == "rq":
        try:
            from redis import Redis
            from rq import Queue

            conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
            conn.ping()
            return RQJobManager(store, Queue("perfusion", connection=conn))
        except Exception:            # rq/redis missing or unreachable -> threads
            pass
    return JobManager(store)


def _info(project: Project) -> ProjectInfo:
    return ProjectInfo(id=project.id, name=project.name,
                       has_vessels=project.has_vessels, has_mask=bool(project.regions),
                       has_result=project.has_result, has_tracer=project.has_tracer)


def create_app(store: ProjectStore | None = None,
               jobs: JobManager | None = None) -> FastAPI:
    """Build the FastAPI app; inject a store/job-manager for testing."""
    from .. import __version__

    store = store or ProjectStore()
    jobs = jobs or _default_jobs(store)
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

    @app.put("/projects/{pid}/io", response_model=ProjectInfo)
    def set_io(pid: str, body: IOIn) -> ProjectInfo:
        """Set arterial inlet / venous outlet points (snapped to vessel ends on run)."""
        project = _project(pid)
        project.inlets = [list(p) for p in body.inlets]
        project.outlets = [list(p) for p in body.outlets]
        store.save(project)
        return _info(project)

    @app.get("/projects/{pid}/io")
    def get_io(pid: str) -> dict:
        project = _project(pid)
        return {"inlets": project.inlets, "outlets": project.outlets}

    @app.post("/projects/{pid}/image", response_model=ProjectInfo)
    async def upload_image(pid: str, request: Request,
                           kind: str = "artery", min_size: int = 20) -> ProjectInfo:
        """Segment an uploaded vessel image into the arterial or venous network.

        The raw image bytes are the request body (any imageio-readable format);
        ``kind`` is ``artery`` (default) or ``vein``. Needs the ``image`` extra.
        The segmented tree is placed in the project's domain and used on the next
        run (like a grown tree).
        """
        if kind not in ("artery", "vein"):
            raise HTTPException(422, "kind must be 'artery' or 'vein'")
        project = _project(pid)
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty request body; POST the image bytes")
        try:
            import imageio.v3 as iio

            from .. import segmentation as seg
        except ImportError:
            raise HTTPException(501, "image segmentation needs the 'image' extra: "
                                     "uv sync --extra image") from None
        try:
            img = iio.imread(io.BytesIO(data))
        except Exception as exc:  # unreadable / non-image payload
            raise HTTPException(422, f"could not decode image: {exc}") from None
        try:
            net = seg.segment_network(img, min_size=min_size)
        except ValueError as exc:  # e.g. no vessel skeleton at this threshold
            raise HTTPException(422, str(exc)) from None
        placed = seg.network_in_domain(net, img.shape, project.grid.domain_limit)
        if kind == "artery":
            project.artery_network = placed
        else:
            project.vein_network = placed
        store.clear_result(project)   # persist the tree; invalidate stale result
        return _info(project)

    @app.post("/projects/{pid}/grow", response_model=GrowInfo)
    def grow_vessels(pid: str, body: GrowRequest | None = None) -> GrowInfo:
        """Grow the arterial tree into the tissue and store it for the next run."""
        project = _project(pid)
        req = body or GrowRequest()
        if req.method not in ("colonization", "cco"):
            raise HTTPException(422, "method must be 'colonization' or 'cco'")
        domain_limit = project.grid.domain_limit
        mask = (rasterize_mask(project.grid.nx, project.grid.ny, domain_limit, project.regions)
                if project.regions else np.ones((project.grid.nx, project.grid.ny), bool))

        base = (runner.polylines_to_network(project.arteries)
                if project.arteries else arterial_tree(project.params.to_core()))
        if req.method == "cco":
            network = growth.cco_grow_in_mask(
                base, mask, domain_limit, n_terminals=req.n_points, seed=req.seed,
                d_min=req.d_min, leaf_radius=req.leaf_radius)
        else:
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
        jid = jobs.submit(project.id, "run", req.model_dump())
        return JobInfo(**jobs.get(jid))

    @app.get("/jobs", response_model=list[JobInfo])
    def list_jobs(project_id: str | None = None) -> list[JobInfo]:
        return [JobInfo(**j) for j in store.list_jobs(project_id=project_id)]

    @app.get("/jobs/{jid}", response_model=JobInfo)
    def job_status(jid: str) -> JobInfo:
        try:
            return JobInfo(**jobs.get(jid))
        except KeyError:
            raise HTTPException(404, f"job {jid!r} not found") from None

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
    def export(pid: str, fmt: str = "npz"):
        from fastapi.responses import FileResponse

        project = _project(pid)
        if not project.has_result:
            raise HTTPException(409, "no result yet; run the simulation first")
        name = project.name or project.id
        if fmt == "npz":       # the stored blob is already the npz export
            return FileResponse(store.result_path(pid), media_type="application/octet-stream",
                                filename=f"{name}.npz")
        if fmt not in ("csv", "vtk"):
            raise HTTPException(422, "fmt must be 'npz', 'csv' or 'vtk'")
        with np.load(store.result_path(pid)) as d:
            dl, mask = tuple(d["domain_limit"]), d["mask"]
            fields = {"pressure": d["pressure"], "flux": d["flux"], "tof": d["tof"]}
        out = Path(tempfile.gettempdir()) / f"perfusion_{pid}.{fmt}"
        if fmt == "csv":
            perfio.fields_to_csv(out, dl, mask=mask, **fields)
            media = "text/csv"
        else:
            perfio.fields_to_vtk(out, dl, mask=mask, **fields)
            media = "application/xml"
        return FileResponse(out, media_type=media, filename=f"{name}.{fmt}")

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
            _add(runner.polylines_to_network(project.arteries), "artery")
        if project.vein_network is not None:
            _add(project.vein_network, "vein")
        elif project.veins:
            _add(runner.polylines_to_network(project.veins), "vein")
        return {"segments": segments}

    @app.post("/projects/{pid}/tracer", response_model=JobInfo)
    def run_tracer(pid: str, body: TracerRequest | None = None) -> JobInfo:
        project = _project(pid)
        req = body or TracerRequest()
        jid = jobs.submit(project.id, "tracer", req.model_dump())
        return JobInfo(**jobs.get(jid))

    @app.get("/projects/{pid}/tracer", response_model=TracerInfo)
    def tracer_frames(pid: str) -> TracerInfo:
        project = _project(pid)
        if not project.has_tracer:
            raise HTTPException(409, "no tracer yet; POST to /tracer first")
        data = store.load_tracer(pid)
        mask, frames = data["mask"], data["frames"]
        # mask inactive cells and serialise NaN -> null
        out = [np.where(mask & np.isfinite(f), f, None).tolist() for f in frames]
        return TracerInfo(id=pid, n_frames=len(frames), shape=mask.shape,
                          times=data["times"].tolist(), vmax=float(data["vmax"]), frames=out)

    @app.get("/projects/{pid}/tracer.gif")
    def tracer_gif(pid: str):
        from fastapi.responses import Response

        project = _project(pid)
        if not project.has_tracer:
            raise HTTPException(409, "no tracer yet; POST to /tracer first")
        try:
            gif = _render_gif(store.load_tracer(pid))
        except ImportError:
            raise HTTPException(501, "GIF export needs the 'image' extra: "
                                     "uv sync --extra image") from None
        return Response(gif, media_type="image/gif",
                        headers={"content-disposition": f'attachment; filename="{pid}.gif"'})

    _mount_frontend(app)
    return app


def _render_gif(tracer: dict, fps: int = 12) -> bytes:
    """Colour-map tracer frames (viridis) into an animated GIF (needs imageio)."""
    import io as _io

    import imageio.v2 as imageio
    from matplotlib import colormaps

    frames, mask, vmax = tracer["frames"], tracer["mask"], float(tracer["vmax"]) or 1.0
    cmap = colormaps["viridis"]
    images = []
    for f in frames:
        rgb = (cmap(np.clip(f / vmax, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
        rgb[~mask] = 245                        # inactive tissue -> light grey
        images.append(np.transpose(rgb, (1, 0, 2))[::-1])   # (nx,ny)->image (row=y down)
    buf = _io.BytesIO()
    imageio.mimsave(buf, images, format="GIF", fps=fps, loop=0)
    return buf.getvalue()


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
