"""Solve + results routes: run/tracer jobs, the job registry, result summary and
fields, exports, the direct-connection render, and the 3-D scenes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from .. import io as perfio
from . import helpers, runner
from .deps import get_jobs, get_project, get_project_2d, get_store
from .schemas import (
    CalibrateRequest,
    FieldResponse,
    JobInfo,
    RunRequest,
    TracerInfo,
    TracerRequest,
    UQRequest,
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
def result_field(pid: str, name: str, project: Project = Depends(get_project_2d),
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


@router.get("/projects/{pid}/kinetics")
def tracer_kinetics(pid: str, model: str = "1CM",
                    project: Project = Depends(get_project),
                    store: ProjectStore = Depends(get_store)) -> dict:
    """Fit a tracer-kinetic model to the computed tracer's tissue curve.

    Takes the spatial-mean tissue concentration C(t) over active cells and the
    stored arterial input, and fits ``model`` (MS / 1CM / 2CUM / 2CXM) to recover
    plasma flow Fp, plasma volume vp, and (2C models) PS / ve. Needs a tracer
    (POST /tracer first)."""
    from ..core import tkm

    if not project.has_tracer:
        raise HTTPException(409, "no tracer yet; POST to /tracer first")
    d = store.load_tracer(pid)
    frames, mask, times = d["frames"], d["mask"], d["times"]
    conc = np.array([float(np.nanmean(f[mask])) if mask.any() else 0.0 for f in frames])
    ca = helpers.tracer_aif(d)
    try:
        r = tkm.fit(times, ca, conc, model=model)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None

    def num(x):
        return float(x) if x is not None and np.isfinite(x) else None

    def arr(a):
        return [num(v) for v in np.asarray(a)] if a is not None else None

    return {
        "model": model.upper(), "Fp": num(r.Fp), "vp": num(r.vp),
        "PS": num(r.PS), "ve": num(r.ve),
        "times": times.tolist(), "aif": arr(ca),
        "measured": arr(conc), "fitted": arr(r.curve),
    }


_SEMI_FIELDS = ("Fp", "vp")
_DECONV_FIELDS = ("Fp", "CBV", "MTT")


@router.get("/projects/{pid}/kinetics/maps", response_model=FieldResponse)
def kinetic_map(pid: str, field: str = "Fp", method: str = "semi",
                oi: float = 0.1, osvd: bool = False,
                project: Project = Depends(get_project),
                store: ProjectStore = Depends(get_store)) -> FieldResponse:
    """Voxel-wise parametric map from the tracer.

    ``method='semi'`` (default) — model-free heuristics: ``field='Fp'`` (max-slope
    plasma flow) or ``'vp'`` (AUC blood volume). ``method='deconv'`` — truncated-SVD
    residue **deconvolution** (Østergaard): ``field`` in ``Fp`` (residue peak), ``CBV``
    (∫residue) or ``MTT`` (CBV/Fp); ``oi`` is the SVD truncation fraction and ``osvd=true``
    uses the block-circulant, delay-insensitive **oSVD** variant. Absolute deconvolution
    accuracy needs the transit time well-sampled (raise the tracer's ``n_frames``);
    relative maps (rCBF/rCBV/rMTT) are robust — the clinical norm."""
    if not project.has_tracer:
        raise HTTPException(409, "no tracer yet; POST to /tracer first")
    allowed = _DECONV_FIELDS if method == "deconv" else _SEMI_FIELDS
    if method not in ("semi", "deconv"):
        raise HTTPException(404, "method must be 'semi' or 'deconv'")
    if field not in allowed:
        raise HTTPException(404, f"field must be one of {allowed} for method='{method}'")
    m = helpers.tracer_map(store.load_tracer(pid), field, method, oi, osvd)
    data = np.where(np.isfinite(m), m, None)
    return FieldResponse(name=field, shape=m.shape, data=data.tolist())


_OXYGEN_FIELDS = ("saturation", "consumption", "concentration")


@router.get("/projects/{pid}/oxygen")
def oxygen_map(pid: str, field: str = "saturation", k: float = 1.0, c_art: float = 1.0,
               diffusivity: float = 0.0, project: Project = Depends(get_project),
               store: ProjectStore = Depends(get_store)) -> dict:
    """Steady oxygen / solute map on the Darcy field (Krogh-style advection–reaction).

    ``field`` in ``saturation`` (C/c_art, ≈1 near feeds → 0 downstream),
    ``consumption`` (local uptake k·C) or ``concentration``. ``k`` is the
    dimensionless Damköhler number (extraction rate relative to perfusion; the global
    OEF ≈ 1−e^{−k} in the well-mixed limit, lower with transit heterogeneity).
    ``diffusivity`` adds tissue diffusion. Hybrid model only — re-solves the field."""
    if not store.result_path(pid).exists():
        raise HTTPException(409, "no field result; the direct-connection model has no grid "
                            "field" if project.has_result else "no result yet; run first")
    if field not in _OXYGEN_FIELDS:
        raise HTTPException(404, f"field must be one of {_OXYGEN_FIELDS}")
    array, oef = runner.oxygen_field(project, field, k=k, c_art=c_art, diffusivity=diffusivity)
    data = np.where(np.isfinite(array), array, None)
    return {"name": field, "shape": list(array.shape), "data": data.tolist(),
            "oef": oef, "k": k}


@router.post("/projects/{pid}/calibrate")
def calibrate_parameter(body: CalibrateRequest, project: Project = Depends(get_project)) -> dict:
    """Inverse problem: recover ``param`` so the model's ``qoi`` matches ``target``
    (bounded 1-D search over forward solves). Useful to fit permeability or inlet
    pressure to a measured flow."""
    try:
        return runner.calibrate(project, body.target, body.param, body.bounds, qoi=body.qoi)
    except runner.InputError as e:
        raise HTTPException(400, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/projects/{pid}/uq")
def uncertainty_quantification(body: UQRequest, project: Project = Depends(get_project)) -> dict:
    """Monte-Carlo uncertainty quantification: sample the given input ranges
    (Latin-hypercube), re-solve, and return each quantity of interest's
    mean/std/5-95 percentiles plus first-order sensitivities (Pearson r). Runs
    ``n_samples`` full solves — keep the grid coarse and the sample count modest."""
    try:
        return runner.uncertainty(project, body.params, n_samples=body.n_samples,
                                  seed=body.seed, method=body.method)
    except runner.InputError as e:
        raise HTTPException(400, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.get("/projects/{pid}/multicompartment")
def multicompartment_map(pid: str, field: str = "perfusion", n: int = 3, beta: float = 1.0,
                         project: Project = Depends(get_project)) -> dict:
    """Multi-compartment Darcy bed (arteriole→…→venule) with inter-compartment
    exchange ``beta``. ``field='perfusion'`` returns the arteriole→next exchange (the
    capillary perfusion map); ``field='comp0'..'comp{n-1}'`` a compartment pressure.
    Re-solves the coupled system — needs a solvable model (vessels or fractal trees)."""
    try:
        array, _ = runner.multicompartment_field(project, field=field, n=n, beta=beta)
    except runner.InputError as e:
        raise HTTPException(400, str(e)) from None
    data = np.where(np.isfinite(array), array, None)
    return {"name": field, "shape": list(array.shape), "data": data.tolist(),
            "n": int(np.clip(n, 2, 5)), "beta": beta}


@router.get("/projects/{pid}/transient")
def transient_response(pid: str, period: float = 1.0, pulse: float = 0.5,
                       compliance: float = 0.02, cycles: int = 4, proximal: float = 0.1,
                       project: Project = Depends(get_project)) -> dict:
    """Pulsatile (Windkessel) response over one cardiac cycle.

    Uses the steady result's inflow for the total resistance R, the vessel
    geometry + ``compliance`` for the total compliance C, then drives a
    ``p_mean ± pulse·ΔP`` sinusoid: the compliance buffers the pulse so tissue
    flow is smoother than the inlet. Needs a solved (hybrid) result."""
    from ..core import transient

    s = project.summary
    if not s or not s.get("arterial_inflow"):
        raise HTTPException(409, "run a simulation first (its steady inflow sets R)")
    p_mean, p_out = float(project.params.p_in), float(project.params.p_out)
    q_steady = float(s["arterial_inflow"])
    if q_steady <= 0 or p_mean <= p_out:
        raise HTTPException(422, "need a positive steady inflow and p_in > p_out")
    r_total = (p_mean - p_out) / q_steady
    # a vessel-less run used the default trees (fitted into the project's domain)
    art, ven = runner.build_networks(project, use_vessels=True, defaults=True)
    c_total = transient.compliance_total([art, ven], compliance)

    cycles = min(max(cycles, 1), 20)
    dt = period / 100.0
    amp = pulse * (p_mean - p_out)
    t, p_in = transient.pulsatile_inlet(p_mean, amp, period, dt, cycles)
    wk = transient.windkessel(p_in, r_total, c_total, dt, p_out=p_out, proximal_fraction=proximal)
    npc = max(1, round(period / dt))
    sl = slice(-npc, None)                              # last cycle = periodic steady state

    def pulsatility(a):
        a = np.asarray(a, float)
        return float((a.max() - a.min()) / max(a.mean(), 1e-12))

    return {
        "R": r_total, "C": c_total, "period": period,
        "t": (t[sl] - t[sl][0]).tolist(), "p_in": p_in[sl].tolist(),
        "P": wk["P"][sl].tolist(), "q_in": wk["q_in"][sl].tolist(),
        "q_tissue": wk["q_tissue"][sl].tolist(),
        "inlet_pulsatility": pulsatility(p_in[sl]),
        "tissue_pulsatility": pulsatility(wk["q_tissue"][sl]),
    }


@router.get("/projects/{pid}/direct")
def direct_result(pid: str, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)) -> dict:
    """Render data for a direct-connection result: arterial/venous vessel
    segments and the terminal-connection links, each ``[x0,y0,x1,y1,flow]``
    (``[x0,y0,z0,x1,y1,z1,flow]`` on a 3-D project, ``dim`` = 3)."""
    if not store._direct_path(pid).exists():
        raise HTTPException(409, "no direct-connection result; run with model='direct'")
    return _direct_payload(store.load_direct(pid))


def _direct_payload(d: dict) -> dict:
    flows = np.concatenate([d["artery"][:, -1], d["vein"][:, -1]])
    return {
        "dim": 3 if d["artery"].shape[1] == 7 else 2,
        "artery": d["artery"].tolist(), "vein": d["vein"].tolist(),
        "connections": d["conn"].tolist(),
        "flow_max": float(flows.max()) if flows.size else 0.0,
    }


@router.get("/projects/{pid}/network-tracer")
def network_tracer(pid: str, project: Project = Depends(get_project),
                   store: ProjectStore = Depends(get_store)) -> dict:
    """Direct-model bolus animation: each vascular element ``[x0,y0,x1,y1]``
    (``[x0,y0,z0,x1,y1,z1]`` in 3-D) with a per-frame concentration (0 arterial /
    1 venous / 2 connection), computed from network transit times. POST /tracer on
    a direct result produces it."""
    if not store._nettracer_path(pid).exists():
        raise HTTPException(409, "no network tracer; run the direct model then POST /tracer")
    return _net_tracer_payload(pid, store.load_network_tracer(pid))


def _net_tracer_payload(pid: str, d: dict) -> dict:
    return {
        "id": pid, "n_frames": int(d["frames"].shape[0]),
        "lines": d["lines"].tolist(), "kind": d["kind"].astype(int).tolist(),
        "frames": d["frames"].astype(float).tolist(), "times": d["times"].tolist(),
        "arrivals": np.where(np.isfinite(d["arrivals"]), d["arrivals"], None).tolist(),
        "vmax": float(d["vmax"]),
    }


@router.get("/projects/{pid}/view3d")
def project_view3d(pid: str, field: str = "pressure", k: float = 1.0, n: int = 3,
                   beta: float = 1.0, project: Project = Depends(get_project),
                   store: ProjectStore = Depends(get_store)) -> dict:
    """This project's result as a 3-D scene. A 2-D project: pressure as a relief
    surface (z ∝ pressure) with the vessel trees at the base. A **3-D project**: its
    3-D trees and a voxel ``field`` — or the tissue as a point cloud before the first
    run. 3-D fields: ``pressure`` / ``flux`` / ``tof`` (the solve), ``saturation`` /
    ``consumption`` (oxygen, Damköhler number ``k``), ``Fp`` / ``vp`` / ``CBV`` /
    ``MTT`` (kinetic maps from the tracer) and ``tracer`` (the bolus animation:
    per-frame voxel values in ``tracer.frames``), and ``mc_perfusion`` /
    ``mc_arteriole`` / ``mc_venule`` (an ``n``-compartment bed with exchange ``beta``:
    the capillary exchange, or the first / last compartment pressure)."""
    if project.grid.is_3d:
        return _view3d_volume(project, store, field, k=k, n=n, beta=beta)
    grid, dom = runner.build_grid(project)
    x0, x1, y0, y1 = dom
    art, ven = runner.build_networks(project, use_vessels=True, defaults=True)

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


_V3_RESULT = ("pressure", "flux", "tof")
_V3_OXYGEN = ("saturation", "consumption")
_V3_TRACER = {"Fp": "semi", "vp": "semi", "CBV": "deconv", "MTT": "deconv"}
_V3_MC = ("mc_perfusion", "mc_arteriole", "mc_venule")
V3_FIELDS = (*_V3_RESULT, *_V3_OXYGEN, *_V3_TRACER, "tracer", *_V3_MC)


def _mc_field(project, field: str, n: int, beta: float):
    """A multi-compartment field for the viewer (``mc_*`` → runner field name)."""
    n = int(np.clip(n, 2, 5))
    name = {"mc_perfusion": "perfusion", "mc_arteriole": "comp0",
            "mc_venule": f"comp{n - 1}"}[field]
    try:
        return runner.multicompartment_field(project, field=name, n=n, beta=beta)[0]
    except runner.InputError as e:
        raise HTTPException(400, str(e)) from None


def _view3d_volume(project, store, field: str, k: float = 1.0, n: int = 3,
                   beta: float = 1.0) -> dict:
    """The 3-D scene of a 3-D project (same payload shape as ``/demo3d``)."""
    if field not in V3_FIELDS:
        raise HTTPException(422, f"field must be one of {list(V3_FIELDS)}")
    grid, dom = runner.build_grid3d(project)
    art, ven = runner.build_networks(project, use_vessels=True, defaults=True)
    segments = [[*map(lambda v: round(v, 2), sg[:6]), round(sg[6], 3), k]
                for net, k in ((art, 0), (ven, 1)) for sg in helpers.segment_lines_3d(net)]
    lo, hi = np.asarray(dom[0::2]), np.asarray(dom[1::2])
    h = (hi - lo) / np.array([grid.nx, grid.ny, grid.nz])
    k = max(1, round((grid.mask.sum() / 3500) ** (1 / 3)))
    idx = np.argwhere(grid.mask[::k, ::k, ::k]) * k
    xyz = np.round(lo + (idx + 0.5) * h, 1)
    out = {"shape": [grid.nx, grid.ny, grid.nz], "domain": [float(v) for v in dom],
           "segments": segments, "has_result": project.has_result, "dim": 3,
           "tissue": project.grid.tissue, "field": field,
           "pressure": {"min": 0.0, "max": 0.0, "voxels": []},
           "tissue_points": xyz.tolist()}
    if store._direct_path(project.id).exists():         # a direct-connection result
        return _view3d_direct(project, store, field, out)
    needs_tracer = field in _V3_TRACER or field == "tracer"
    if needs_tracer and not project.has_tracer:
        raise HTTPException(409, "no tracer yet; run the flow (POST /tracer) first")
    if project.has_result and field == "tracer":
        d = store.load_tracer(project.id)
        frames = d["frames"][:, idx[:, 0], idx[:, 1], idx[:, 2]]       # (n_t, n_vox)
        vmax = float(d["vmax"]) or 1.0
        out["pressure"] = {"min": 0.0, "max": vmax, "name": field, "voxels": []}
        out["tracer"] = {"xyz": xyz.tolist(), "times": d["times"].tolist(), "vmax": vmax,
                         "frames": np.round(np.nan_to_num(frames), 4).tolist()}
        out["summary"] = project.summary
    elif project.has_result:
        if field in _V3_RESULT:
            f = store.result_field(project.id, field)
        elif field in _V3_OXYGEN:
            f, oef = runner.oxygen_field(project, field, k=k)
            out["oef"] = oef
        elif field in _V3_MC:
            f = _mc_field(project, field, n, beta)
            out["mc"] = {"n": int(np.clip(n, 2, 5)), "beta": beta,
                         "perfusion": float(np.nansum(f)) if field == "mc_perfusion" else None}
        else:
            f = helpers.tracer_map(store.load_tracer(project.id), field, _V3_TRACER[field])
        vals = f[tuple(idx.T)]
        ok = np.isfinite(vals)
        fin = f[grid.mask & np.isfinite(f)]
        lo_v, hi_v = (np.percentile(fin, [2, 98]) if fin.size else (0.0, 1.0))
        out["pressure"] = {"min": float(lo_v), "max": float(hi_v), "name": field,
                           "voxels": np.column_stack([xyz[ok], np.round(vals[ok], 6)]).tolist()}
        out["summary"] = project.summary
    return out


def _view3d_direct(project, store, field: str, out: dict) -> dict:
    """A 3-D direct-connection result: no tissue field — the trees and their resistive
    terminal links coloured by flow (``direct``) and, after POST /tracer, the bolus
    timed along the network (``net_tracer``: per-element frames)."""
    if field not in ("pressure", "tracer"):
        raise HTTPException(409, "the direct-connection model has no tissue field; "
                                 "re-run with the hybrid (Darcy) model for this map")
    out["direct"] = _direct_payload(store.load_direct(project.id))
    out["summary"] = project.summary
    if field == "tracer":
        if not store._nettracer_path(project.id).exists():
            raise HTTPException(409, "no tracer yet; run the flow (POST /tracer) first")
        out["net_tracer"] = _net_tracer_payload(project.id,
                                                store.load_network_tracer(project.id))
    return out


@router.get("/projects/{pid}/export")
def export(pid: str, fmt: str = "npz", project: Project = Depends(get_project),
           store: ProjectStore = Depends(get_store)):
    from fastapi.responses import FileResponse

    name = project.name or project.id
    if fmt == "vtp":       # vessel-network PolyData (tubes) — no solved field needed
        net, _ = runner.build_networks(project, use_vessels=True, defaults=True)
        out = Path(tempfile.gettempdir()) / f"perfusion_{pid}.vtp"
        perfio.network_to_vtp(net, out)
        return FileResponse(out, media_type="application/xml", filename=f"{name}.vtp")
    _require_field_result(project, store)
    if fmt == "npz":       # the stored blob is already the npz export
        return FileResponse(store.result_path(pid), media_type="application/octet-stream",
                            filename=f"{name}.npz")
    if fmt not in ("csv", "vtk"):
        raise HTTPException(422, "fmt must be 'npz', 'csv', 'vtk' or 'vtp'")
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


def _require_field_result(project, store) -> None:
    """409 unless the project holds a solved *field* (the hybrid result blob)."""
    if not project.has_result:
        raise HTTPException(409, "no result yet; run the simulation first")
    if not store.result_path(project.id).exists():
        raise HTTPException(409, "the direct-connection model has no tissue field to "
                                 "export; export the vessels (fmt=vtp) or re-run the "
                                 "hybrid model")


@router.get("/projects/{pid}/bundle")
def export_bundle(pid: str, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)):
    """One zip with everything: npz + CSV + VTK fields (2-D or 3-D), summary/meta
    JSON, and — if a tracer exists — its frames (``tracer.npz``) plus, in 2-D, the
    tracer GIF / MP4."""
    import contextlib
    import json
    import zipfile

    from fastapi.responses import FileResponse

    _require_field_result(project, store)
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
            z.write(store.tracer_path(pid), "tracer.npz")
        if project.has_tracer and not project.grid.is_3d:     # GIF / MP4 are 2-D renders
            tracer = store.load_tracer(pid)
            with contextlib.suppress(ImportError):   # GIF needs the image extra
                z.writestr("tracer.gif", helpers.render_gif(tracer))
            with contextlib.suppress(ImportError):   # MP4 needs the video extra
                z.writestr("tracer.mp4", helpers.render_mp4(tracer))
    return FileResponse(out, media_type="application/zip", filename=f"{name}.zip")


@router.get("/projects/{pid}/tracer", response_model=TracerInfo)
def tracer_frames(pid: str, project: Project = Depends(get_project_2d),
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
def tracer_gif(pid: str, project: Project = Depends(get_project_2d),
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


@router.get("/projects/{pid}/tracer.mp4")
def tracer_mp4(pid: str, fps: int = 12, project: Project = Depends(get_project_2d),
               store: ProjectStore = Depends(get_store)):
    """The tracer flow video as an H.264 MP4 (server-side ffmpeg)."""
    from fastapi.responses import Response

    if not project.has_tracer:
        raise HTTPException(409, "no tracer yet; POST to /tracer first")
    try:
        mp4 = helpers.render_mp4(store.load_tracer(pid), fps=int(np.clip(fps, 1, 60)))
    except ImportError:
        raise HTTPException(501, "MP4 export needs the 'video' extra: "
                                 "uv sync --extra video") from None
    return Response(mp4, media_type="video/mp4",
                    headers={"content-disposition": f'attachment; filename="{pid}.mp4"'})
