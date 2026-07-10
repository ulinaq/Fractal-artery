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
    GridIn,
    GrowInfo,
    GrowRequest,
    IOIn,
    JobInfo,
    MaskIn,
    NetworkModel,
    NetworksEditIn,
    ParametersIn,
    ProjectCreate,
    ProjectDocument,
    ProjectInfo,
    RunRequest,
    TracerInfo,
    TracerRequest,
    VesselsIn,
)
from .store import (
    Project,
    ProjectStore,
    document_from_project,
    project_from_document,
)

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
    has_mask = bool(project.regions) or getattr(project, "mask", None) is not None
    return ProjectInfo(id=project.id, name=project.name,
                       has_vessels=project.has_vessels, has_mask=has_mask,
                       has_result=project.has_result, has_tracer=project.has_tracer,
                       params=project.params, grid=project.grid)


def _frog_networks_and_mask(nx: int, ny: int, dom):
    """Segment the bundled frog system into (artery, vein, raster mask) placed in
    ``dom``. Raises ImportError without the ``image`` extra."""
    from .. import segmentation as seg

    system = seg.load_frog_system()
    return (seg.network_in_domain(system.artery, system.mask.shape, dom),
            seg.network_in_domain(system.vein, system.mask.shape, dom),
            seg.mask_in_domain(system.mask, nx, ny, dom))


_DEMO3D_CACHE: dict = {}


def _demo3d_frog() -> dict:
    """The 2-D frog result as a 3-D pressure-relief surface (z = pressure) with
    the vasculature at the base — for the 3-D viewer."""
    if "frog" in _DEMO3D_CACHE:
        return _DEMO3D_CACHE["frog"]
    from ..core.darcy import Grid
    from ..core.parameters import Parameters
    from ..core.pipeline import run_hybrid

    dom = (0.0, 100.0, 0.0, 120.0)
    nx, ny = 120, 140
    art, ven, mask = _frog_networks_and_mask(nx, ny, dom)
    grid = Grid(nx=nx, ny=ny, lx=100.0, ly=120.0, mask=mask)
    res = run_hybrid(Parameters(p_in=40.0, p_out=0.0), grid, domain_limit=dom, art=art, ven=ven)
    p = res.pressure
    finite = p[np.isfinite(p) & mask]
    pmin, pmax = float(finite.min()), float(finite.max())
    span = (pmax - pmin) or 1.0
    relief = 35.0
    xc = (np.arange(nx) + 0.5) / nx * 100.0
    yc = (np.arange(ny) + 0.5) / ny * 120.0
    stride = max(1, round((nx * ny / 2500) ** 0.5))
    voxels = [
        [round(float(xc[ix]), 1), round(float(yc[iy]), 1),
         round((p[ix, iy] - pmin) / span * relief, 2), round(float(p[ix, iy]), 4)]
        for ix in range(0, nx, stride) for iy in range(0, ny, stride)
        if mask[ix, iy] and np.isfinite(p[ix, iy])
    ]

    def segs(net, kind):
        return [[round(float(net.node_x[net.seg_start[s]]), 2),
                 round(float(net.node_y[net.seg_start[s]]), 2), 0.0,
                 round(float(net.node_x[net.seg_end[s]]), 2),
                 round(float(net.node_y[net.seg_end[s]]), 2), 0.0,
                 round(float(net.seg_radius[s]), 2), 0 if kind == "artery" else 1]
                for s in range(net.n_segments)]

    payload = {"domain": [0.0, 100.0, 0.0, 120.0, 0.0, relief], "shape": [nx, ny, 1],
               "pressure": {"min": pmin, "max": pmax, "voxels": voxels},
               "segments": segs(art, "artery") + segs(ven, "vein")}
    _DEMO3D_CACHE["frog"] = payload
    return payload


def _demo3d_payload(gen: int = 6) -> dict:
    """Run a default 3-D hybrid and serialise trees + a subsampled voxel field.

    Cached per ``gen``. Feeds the browser 3-D viewer (``frontend/viewer3d.js``).
    """
    gen = int(np.clip(gen, 3, 7))
    if gen in _DEMO3D_CACHE:
        return _DEMO3D_CACHE[gen]

    from ..core.darcy3d import Grid3D
    from ..core.geometry import grow_tree_3d
    from ..core.parameters import Parameters
    from ..core.pipeline3d import run_hybrid_3d

    art = grow_tree_3d((25, 25, 3), (0, 0, 1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=gen)
    ven = grow_tree_3d((25, 25, 47), (0, 0, -1), root_radius=2.0, min_radius=0.8,
                       length_to_radius=5, ratio=0.8, max_generations=gen, mirror=True)
    xs = np.concatenate([art.node_x, ven.node_x])
    ys = np.concatenate([art.node_y, ven.node_y])
    zs = np.concatenate([art.node_z, ven.node_z])
    m = 4.0
    dom = (xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m, zs.min() - m, zs.max() + m)
    grid = Grid3D(nx=26, ny=26, nz=28,
                  lx=dom[1] - dom[0], ly=dom[3] - dom[2], lz=dom[5] - dom[4])
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)

    def segments(net, kind):
        out = []
        for s in range(net.n_segments):
            a, b = int(net.seg_start[s]), int(net.seg_end[s])
            out.append([round(float(net.node_x[a]), 2), round(float(net.node_y[a]), 2),
                        round(float(net.node_z[a]), 2), round(float(net.node_x[b]), 2),
                        round(float(net.node_y[b]), 2), round(float(net.node_z[b]), 2),
                        round(float(net.seg_radius[s]), 2), 0 if kind == "artery" else 1])
        return out

    p, mask = res.pressure, grid.mask
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    finite = p[np.isfinite(p) & mask]
    pmin, pmax = float(finite.min()), float(finite.max())
    stride = max(1, round((nx * ny * nz / 3000) ** (1 / 3)))
    xmin, _, ymin, _, zmin, _ = dom
    voxels = [
        [round(xmin + (ix + 0.5) * grid.hx, 1), round(ymin + (iy + 0.5) * grid.hy, 1),
         round(zmin + (iz + 0.5) * grid.hz, 1), round(float(p[ix, iy, iz]), 4)]
        for ix in range(0, nx, stride)
        for iy in range(0, ny, stride)
        for iz in range(0, nz, stride)
        if mask[ix, iy, iz] and np.isfinite(p[ix, iy, iz])
    ]
    payload = {
        "domain": [round(v, 1) for v in dom], "shape": [nx, ny, nz],
        "pressure": {"min": pmin, "max": pmax, "voxels": voxels},
        "segments": segments(art, "artery") + segments(ven, "vein"),
    }
    _DEMO3D_CACHE[gen] = payload
    return payload


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

    @app.get("/projects/{pid}/document")
    def export_document(pid: str):
        """Download the whole project as one portable JSON file (save your work)."""
        from fastapi.responses import JSONResponse

        project = _project(pid)
        doc = document_from_project(project)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name) or "project"
        return JSONResponse(
            doc.model_dump(mode="json"),
            headers={"Content-Disposition": f'attachment; filename="{safe}.perfusion.json"'},
        )

    @app.post("/projects/import", response_model=ProjectInfo)
    def import_document(doc: ProjectDocument) -> ProjectInfo:
        """Restore a saved project document as a *new* project (continue later)."""
        import uuid

        project = project_from_document(doc, uuid.uuid4().hex[:12])
        store.save(project)
        return _info(project)

    @app.get("/projects/{pid}/editable")
    def get_editable(pid: str) -> dict:
        """Node-level geometry of the arterial/venous trees, for interactive edits.

        Returns each 2-D tree as a :class:`NetworkModel` (nodes + segments + radii),
        or ``null`` when the tree is absent or 3-D (the 2-D editor cannot move it).
        """
        project = _project(pid)

        def enc(net):
            if net is None or net.is_3d:
                return None
            return NetworkModel.from_core(net).model_dump()

        return {"artery": enc(project.artery_network), "vein": enc(project.vein_network)}

    @app.put("/projects/{pid}/editable", response_model=ProjectInfo)
    def put_editable(pid: str, body: NetworksEditIn) -> ProjectInfo:
        """Apply moved nodes + resized segments to the stored trees (topology kept).

        Segment lengths/angles are recomputed from the new node positions; the
        result is invalidated because the geometry changed.
        """
        import dataclasses

        project = _project(pid)

        def apply(net, edit):
            nx_ = np.asarray(edit.node_x, float)
            ny_ = np.asarray(edit.node_y, float)
            radius = np.maximum(np.asarray(edit.seg_radius, float), 1e-3)
            if len(nx_) != net.n_nodes or len(ny_) != net.n_nodes:
                raise HTTPException(422, f"expected {net.n_nodes} nodes")
            if len(radius) != net.n_segments:
                raise HTTPException(422, f"expected {net.n_segments} segment radii")
            ss, se = net.seg_start, net.seg_end
            return dataclasses.replace(
                net, node_x=nx_, node_y=ny_, seg_radius=radius,
                seg_length=np.hypot(nx_[se] - nx_[ss], ny_[se] - ny_[ss]),
                seg_angle=np.arctan2(ny_[se] - ny_[ss], nx_[se] - nx_[ss]),
            )

        changed = False
        if body.artery is not None and project.artery_network is not None:
            project.artery_network = apply(project.artery_network, body.artery)
            changed = True
        if body.vein is not None and project.vein_network is not None:
            project.vein_network = apply(project.vein_network, body.vein)
            changed = True
        if not changed:
            raise HTTPException(404, "project has no editable arterial/venous tree")
        store.clear_result(project)     # geometry changed -> stale result; also saves
        return _info(project)

    @app.post("/projects/frog", response_model=ProjectInfo)
    def create_frog(name: str = "frog tongue") -> ProjectInfo:
        """One-click example: the segmented frog-tongue vasculature + organ mask.

        Segments the bundled frog artery/vein images and tongue mask and places
        them in the app's standard domain (reusing ``segmentation.network_in_domain``
        / ``mask_in_domain``) — a real coupled system ready to Run. Needs the
        ``image`` extra.
        """
        dom = (0.0, 100.0, 0.0, 120.0)
        nx, ny = 120, 140
        try:
            art, ven, mask = _frog_networks_and_mask(nx, ny, dom)
        except ImportError:
            raise HTTPException(501, "the frog example needs the 'image' extra: "
                                     "uv sync --extra image") from None
        grid = GridIn(nx=nx, ny=ny, lx=100.0, ly=120.0, domain_limit=dom)
        project = store.create(name, ParametersIn(p_in=40.0, p_out=0.0), grid)
        project.artery_network, project.vein_network, project.mask = art, ven, mask
        # Tongue outline as a display polygon (shows/edits in Tissue mode). The raster
        # mask above still drives the solve; this is aligned to it for visualisation.
        from skimage.measure import find_contours

        outline = max(find_contours(mask.astype(float), 0.5), key=len)
        step = max(1, len(outline) // 80)
        project.regions = [[[round(float(r) / nx * 100.0, 2), round(float(c) / ny * 120.0, 2)]
                            for r, c in outline[::step]]]
        store.save(project)
        return _info(project)

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

    @app.get("/projects/{pid}/bundle")
    def export_bundle(pid: str):
        """One zip with everything: npz + CSV + VTK fields, summary/meta JSON, and
        the tracer GIF if one exists."""
        import contextlib
        import json
        import zipfile

        from fastapi.responses import FileResponse

        project = _project(pid)
        if not project.has_result:
            raise HTTPException(409, "no result yet; run the simulation first")
        name = project.name or project.id
        with np.load(store.result_path(pid)) as d:
            dl, mask = tuple(d["domain_limit"]), d["mask"]
            fields = {"pressure": d["pressure"], "flux": d["flux"], "tof": d["tof"]}
        tmp = Path(tempfile.gettempdir())
        csv_path, vtk_path = tmp / f"{pid}_f.csv", tmp / f"{pid}_f.vti"
        perfio.fields_to_csv(csv_path, dl, mask=mask, **fields)
        perfio.fields_to_vtk(vtk_path, dl, mask=mask, **fields)
        meta = {"id": pid, "name": project.name,
                "params": project.params.model_dump(), "grid": project.grid.model_dump(),
                "inlets": project.inlets, "outlets": project.outlets,
                "regions": project.regions}

        out = tmp / f"perfusion_bundle_{pid}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(store.result_path(pid), "results.npz")
            z.write(csv_path, "fields.csv")
            z.write(vtk_path, "fields.vti")
            z.writestr("summary.json", json.dumps(project.summary, indent=2))
            z.writestr("meta.json", json.dumps(meta, indent=2))
            if project.has_tracer:
                with contextlib.suppress(ImportError):   # GIF needs the image extra
                    z.writestr("tracer.gif", _render_gif(store.load_tracer(pid)))
        return FileResponse(out, media_type="application/zip", filename=f"{name}.zip")

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

    @app.get("/demo3d")
    def demo3d(gen: int = 6, model: str = "synthetic") -> dict:
        """A ready-to-render result for the 3-D viewer.

        ``model=synthetic`` (default) is a 3-D fractal-tree hybrid; ``model=frog``
        is the real 2-D frog result shown as a 3-D pressure-relief surface.
        """
        if model == "frog":
            try:
                return _demo3d_frog()
            except ImportError:
                raise HTTPException(501, "the frog model needs the 'image' extra: "
                                         "uv sync --extra image") from None
        return _demo3d_payload(gen)

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
    if not (frontend / "index.html").exists():
        return
    from fastapi.staticfiles import StaticFiles

    class _NoCacheStatic(StaticFiles):
        # Force the browser to revalidate (ETag) every load, so an updated app.js/
        # index.html is never served stale from cache.
        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    # Mounted last, so the API routes above take precedence over "/".
    app.mount("/", _NoCacheStatic(directory=frontend, html=True), name="frontend")


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
