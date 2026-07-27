"""Solve + results routes: run/tracer jobs, the job registry, result summary and
fields, exports, the direct-connection render, and the 3-D scenes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .. import io as perfio
from . import helpers, runner
from .deps import get_jobs, get_project, get_store
from .schemas import (
    FieldResponse,
    JobInfo,
    RunRequest,
    TracerInfo,
    TracerRequest,
)
from .store import Project, ProjectStore

router = APIRouter()


@router.post("/projects/{pid}/run", response_model=JobInfo)
def run_project(body: RunRequest | None = None, project: Project = Depends(get_project),
                jobs=Depends(get_jobs)) -> JobInfo:
    req = body or RunRequest()
    jid = jobs.submit(project.id, "run", req.model_dump())
    return JobInfo(**jobs.get(jid))


@router.post("/projects/{pid}/tracer", response_model=JobInfo)
def run_tracer(body: TracerRequest | None = None, project: Project = Depends(get_project),
               jobs=Depends(get_jobs)) -> JobInfo:
    req = body or TracerRequest()
    jid = jobs.submit(project.id, "tracer", req.model_dump())
    return JobInfo(**jobs.get(jid))


@router.get("/jobs", response_model=list[JobInfo])
def list_jobs(project_id: str | None = None,
              store: ProjectStore = Depends(get_store)) -> list[JobInfo]:
    return [JobInfo(**j) for j in store.list_jobs(project_id=project_id)]


@router.get("/jobs/{jid}", response_model=JobInfo)
def job_status(jid: str, jobs=Depends(get_jobs)) -> JobInfo:
    try:
        return JobInfo(**jobs.get(jid))
    except KeyError:
        raise HTTPException(404, f"job {jid!r} not found") from None


@router.get("/projects/{pid}/results")
def results(project: Project = Depends(get_project)) -> dict:
    if project.summary is None:
        raise HTTPException(409, "no result yet; run the simulation first")
    return project.summary


@router.get("/projects/{pid}/results/field/{name}", response_model=FieldResponse)
def result_field(pid: str, name: str, project: Project = Depends(get_project),
                 store: ProjectStore = Depends(get_store)) -> FieldResponse:
    if not store.result_path(pid).exists():
        raise HTTPException(409, "no field result; the direct-connection model has no "
                            "grid field — use GET /projects/{id}/direct" if project.has_result
                            else "no result yet; run the simulation first")
    if name not in helpers.FIELDS:
        raise HTTPException(404, f"unknown field {name!r}; use {list(helpers.FIELDS)}")
    field = store.result_field(pid, helpers.FIELDS[name])
    mask = store.result_field(pid, "mask")
    field = np.where(mask, field, np.nan)             # inactive cells -> null
    data = np.where(np.isfinite(field), field, None)  # NaN -> null (valid JSON)
    return FieldResponse(name=name, shape=field.shape, data=data.tolist())


@router.get("/projects/{pid}/transit")
def transit_distribution(pid: str, bins: int = 40, k: float = 0.0,
                         project: Project = Depends(get_project),
                         store: ProjectStore = Depends(get_store)) -> dict:
    """Transit-time distribution of the solved field: MTT, CTH (heterogeneity),
    relative dispersion, a Kety-model oxygen-extraction fraction, and h(τ).

    ``k`` is the extraction rate; ``k <= 0`` (default) auto-scales it to ``1/MTT``,
    giving a unit-free heterogeneity index (≈0.63 for homogeneous flow, lower as
    CTH grows). Pass a positive ``k`` for a fixed physical rate.
    """
    from ..core.perfusion_metrics import (
        oxygen_extraction_fraction,
        transit_histogram,
        transit_metrics,
    )

    if not store.result_path(pid).exists():
        raise HTTPException(409, "no transit-time field; run the hybrid model first "
                            "(the direct-connection model has no grid field)")
    bins = int(np.clip(bins, 5, 200))
    tof = np.where(store.result_field(pid, "mask"), store.result_field(pid, "tof"), np.nan)
    flux = store.result_field(pid, "flux")
    tm = transit_metrics(tof, weights=flux)
    kk = k if k > 0 else (1.0 / tm.mtt if tm.mtt > 0 else 1.0)
    centres, density = transit_histogram(tof, weights=flux, bins=bins)
    return {
        "mtt": tm.mtt, "cth": tm.cth, "rd": tm.rd, "n": tm.n,
        "oef": oxygen_extraction_fraction(tof, weights=flux, k=kk), "k": kk,
        "histogram": {"centres": centres.tolist(), "density": density.tolist()},
    }


@router.get("/projects/{pid}/direct")
def direct_result(pid: str, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)) -> dict:
    """Render data for a direct-connection result: arterial/venous vessel
    segments and the terminal-connection links, each ``[x0,y0,x1,y1,flow]``."""
    if not store._direct_path(pid).exists():
        raise HTTPException(409, "no direct-connection result; run with model='direct'")
    d = store.load_direct(pid)
    flows = (np.concatenate([d["artery"][:, 4], d["vein"][:, 4]])
             if d["artery"].size else np.array([0.0]))
    return {
        "artery": d["artery"].tolist(), "vein": d["vein"].tolist(),
        "connections": d["conn"].tolist(),
        "flow_max": float(flows.max()) if flows.size else 0.0,
    }


@router.get("/projects/{pid}/view3d")
def project_view3d(pid: str, project: Project = Depends(get_project),
                   store: ProjectStore = Depends(get_store)) -> dict:
    """This project's result as a 3-D scene: pressure as a relief surface
    (z ∝ pressure) with the vessel trees at the base. Voxels only once solved."""
    grid, dom = runner.build_grid(project)
    x0, x1, y0, y1 = dom
    art, ven = runner.build_networks(project, use_vessels=True)

    def segs(net, kind):
        if net is None:
            return []
        return [[x0, y0, 0.0, x1, y1, 0.0, round(r, 3), kind]
                for x0, y0, x1, y1, r in helpers.segment_lines(net)]

    segments = segs(art, 0) + segs(ven, 1)
    voxels, pmin, pmax, relief = [], 0.0, 0.0, 35.0
    zmax = 1.0
    if project.has_result:
        p = store.result_field(pid, "pressure")
        mask = grid.mask if grid.mask is not None else np.ones((grid.nx, grid.ny), bool)
        finite = np.isfinite(p) & mask
        if finite.any():
            vals = p[finite]
            pmin, pmax = float(vals.min()), float(vals.max())
            span = (pmax - pmin) or 1.0
            xc = (np.arange(grid.nx) + 0.5) / grid.nx * (x1 - x0) + x0
            yc = (np.arange(grid.ny) + 0.5) / grid.ny * (y1 - y0) + y0
            stride = max(1, round((grid.nx * grid.ny / 2500) ** 0.5))
            voxels = [
                [round(float(xc[ix]), 1), round(float(yc[iy]), 1),
                 round((p[ix, iy] - pmin) / span * relief, 2), round(float(p[ix, iy]), 4)]
                for ix in range(0, grid.nx, stride) for iy in range(0, grid.ny, stride)
                if finite[ix, iy]
            ]
            zmax = relief
    return {"shape": [grid.nx, grid.ny, 1], "domain": [x0, x1, y0, y1, 0.0, zmax],
            "segments": segments, "has_result": project.has_result,
            "pressure": {"min": pmin, "max": pmax, "voxels": voxels}}


@router.get("/projects/{pid}/export")
def export(pid: str, fmt: str = "npz", project: Project = Depends(get_project),
           store: ProjectStore = Depends(get_store)):
    from fastapi.responses import FileResponse

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


@router.get("/projects/{pid}/bundle")
def export_bundle(pid: str, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)):
    """One zip with everything: npz + CSV + VTK fields, summary/meta JSON, and
    the tracer GIF if one exists."""
    import contextlib
    import json
    import zipfile

    from fastapi.responses import FileResponse

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
                z.writestr("tracer.gif", helpers.render_gif(store.load_tracer(pid)))
    return FileResponse(out, media_type="application/zip", filename=f"{name}.zip")


@router.get("/projects/{pid}/tracer", response_model=TracerInfo)
def tracer_frames(pid: str, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)) -> TracerInfo:
    if not project.has_tracer:
        raise HTTPException(409, "no tracer yet; POST to /tracer first")
    data = store.load_tracer(pid)
    mask, frames = data["mask"], data["frames"]
    # mask inactive cells and serialise NaN -> null
    out = [np.where(mask & np.isfinite(f), f, None).tolist() for f in frames]
    return TracerInfo(id=pid, n_frames=len(frames), shape=mask.shape,
                      times=data["times"].tolist(), vmax=float(data["vmax"]), frames=out)


@router.get("/projects/{pid}/tracer.gif")
def tracer_gif(pid: str, project: Project = Depends(get_project),
               store: ProjectStore = Depends(get_store)):
    from fastapi.responses import Response

    if not project.has_tracer:
        raise HTTPException(409, "no tracer yet; POST to /tracer first")
    try:
        gif = helpers.render_gif(store.load_tracer(pid))
    except ImportError:
        raise HTTPException(501, "GIF export needs the 'image' extra: "
                                 "uv sync --extra image") from None
    return Response(gif, media_type="image/gif",
                    headers={"content-disposition": f'attachment; filename="{pid}.gif"'})


@router.get("/demo3d")
def demo3d(gen: int = 6, model: str = "synthetic") -> dict:
    """A ready-to-render result for the 3-D viewer.

    ``model=synthetic`` (default) is a 3-D fractal-tree hybrid; ``model=frog``
    is the real 2-D frog result shown as a 3-D pressure-relief surface.
    """
    if model == "frog":
        try:
            return helpers.demo3d_frog()
        except ImportError:
            raise HTTPException(501, "the frog model needs the 'image' extra: "
                                     "uv sync --extra image") from None
    return helpers.demo3d_payload(gen)
