"""Vessel/tissue geometry routes: hand-drawn vessels, tissue mask, inlet/outlet
points, interactive network editing, vectorising strokes, image segmentation,
tree growth, and the overlay geometry."""

from __future__ import annotations

import dataclasses
import io
from itertools import pairwise

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import growth
from ..core.geometry import arterial_tree, rasterize_mask
from . import helpers, runner
from .deps import get_project, get_store, project_info
from .schemas import (
    GrowInfo,
    GrowRequest,
    IOIn,
    MaskIn,
    NetworkModel,
    NetworksEditIn,
    ProjectInfo,
    VesselsIn,
)
from .store import Project, ProjectStore

router = APIRouter()


@router.get("/projects/{pid}/vessels", response_model=VesselsIn)
def get_vessels(project: Project = Depends(get_project)) -> VesselsIn:
    return VesselsIn(arteries=project.arteries, veins=project.veins)


@router.put("/projects/{pid}/vessels", response_model=ProjectInfo)
def set_vessels(body: VesselsIn, project: Project = Depends(get_project),
                store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    project.arteries, project.veins = body.arteries, body.veins
    store.save(project)
    return project_info(project)


@router.get("/projects/{pid}/mask")
def get_mask(project: Project = Depends(get_project)) -> dict:
    return {"regions": project.regions}


@router.put("/projects/{pid}/mask", response_model=ProjectInfo)
def set_mask(body: MaskIn, project: Project = Depends(get_project),
             store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    project.regions = [[list(pt) for pt in poly] for poly in body.regions]
    store.save(project)
    return project_info(project)


@router.get("/projects/{pid}/io")
def get_io(project: Project = Depends(get_project)) -> dict:
    return {"inlets": project.inlets, "outlets": project.outlets}


@router.put("/projects/{pid}/io", response_model=ProjectInfo)
def set_io(body: IOIn, project: Project = Depends(get_project),
           store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Set arterial inlet / venous outlet points (snapped to vessel ends on run)."""
    project.inlets = [list(p) for p in body.inlets]
    project.outlets = [list(p) for p in body.outlets]
    store.save(project)
    return project_info(project)


@router.post("/projects/{pid}/vectorize", response_model=ProjectInfo)
def vectorize(project: Project = Depends(get_project),
              store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Turn hand-drawn polylines into an editable vessel **network** (tree/graph).

    Vessels are a network, not free polylines: each drawn stroke is exploded
    into consecutive 2-point edges (so bends stay as degree-2 nodes and each
    edge carries its own radius), shared endpoints snap into junctions. The
    result is stored as ``artery_network``/``vein_network`` and the polylines
    are cleared, so editing (and the solver) see one consistent graph.
    """
    from ..core.geometry import network_from_polylines

    def to_network(existing, polylines):
        if existing is not None or not polylines:
            return existing, polylines        # already a network, or nothing to do
        segs, radii = [], []
        for p in polylines:
            pts = [list(pt) for pt in p.points]
            for a, b in pairwise(pts):
                segs.append([a, b])
                radii.append(p.radius)
        if not segs:
            return existing, polylines
        return network_from_polylines(segs, radii=radii, snap_tol=0.5), []

    art_net, art_poly = to_network(project.artery_network, project.arteries)
    ven_net, ven_poly = to_network(project.vein_network, project.veins)
    project.artery_network, project.arteries = art_net, art_poly
    project.vein_network, project.veins = ven_net, ven_poly
    if project.artery_network is None and project.vein_network is None:
        raise HTTPException(400, "no vessels to convert to a network")
    store.clear_result(project)     # geometry model changed -> stale result; saves
    return project_info(project)


@router.get("/projects/{pid}/editable")
def get_editable(project: Project = Depends(get_project)) -> dict:
    """Node-level geometry of the arterial/venous trees, for interactive edits.

    Returns each 2-D tree as a :class:`NetworkModel` (nodes + segments + radii),
    or ``null`` when the tree is absent or 3-D (the 2-D editor cannot move it).
    """
    def enc(net):
        if net is None or net.is_3d:
            return None
        return NetworkModel.from_core(net).model_dump()

    return {"artery": enc(project.artery_network), "vein": enc(project.vein_network)}


@router.put("/projects/{pid}/editable", response_model=ProjectInfo)
def put_editable(body: NetworksEditIn, project: Project = Depends(get_project),
                 store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Apply moved nodes + resized segments to the stored trees.

    A pure move/resize (no ``seg_start``/``seg_end``) keeps the topology; sending
    topology rebuilds it. Either way the result is invalidated (geometry changed).
    """
    def apply(net, edit):
        if edit.seg_start is not None and edit.seg_end is not None:
            # topology may have changed (add/delete) -> rebuild from scratch
            return helpers.rebuild_network(net, edit.node_x, edit.node_y, edit.seg_start,
                                           edit.seg_end, edit.seg_radius,
                                           edit.inlet_nodes, edit.root_node)
        # pure move/resize -> keep topology, just refresh geometry
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
    return project_info(project)


@router.post("/projects/{pid}/image", response_model=ProjectInfo)
async def upload_image(request: Request, kind: str = "artery", min_size: int = 20,
                       project: Project = Depends(get_project),
                       store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Segment an uploaded vessel image into the arterial or venous network.

    The raw image bytes are the request body (any imageio-readable format);
    ``kind`` is ``artery`` (default) or ``vein``. Needs the ``image`` extra.
    """
    if kind not in ("artery", "vein"):
        raise HTTPException(422, "kind must be 'artery' or 'vein'")
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
    return project_info(project)


@router.post("/projects/{pid}/grow", response_model=GrowInfo)
def grow_vessels(body: GrowRequest | None = None, project: Project = Depends(get_project),
                 store: ProjectStore = Depends(get_store)) -> GrowInfo:
    """Grow the arterial *or* venous tree into the tissue (``target``).

    The base is the drawn/segmented vessels of that tree (shape preserved, so
    branches extend the existing vessels rather than a straight trunk), else
    the default fractal tree. Artery and vein grow independently.
    """
    from ..core.geometry import venous_tree

    req = body or GrowRequest()
    if req.method not in ("colonization", "cco"):
        raise HTTPException(422, "method must be 'colonization' or 'cco'")
    if req.target not in ("artery", "vein"):
        raise HTTPException(422, "target must be 'artery' or 'vein'")
    domain_limit = project.grid.domain_limit
    mask = (rasterize_mask(project.grid.nx, project.grid.ny, domain_limit, project.regions)
            if project.regions else np.ones((project.grid.nx, project.grid.ny), bool))

    prm = project.params.to_core()
    if req.target == "artery":
        polys, existing, default = project.arteries, project.artery_network, arterial_tree(prm)
    else:
        polys, existing, default = project.veins, project.vein_network, venous_tree(prm)
    # prefer the drawn strokes (bends kept) so growth extends them, else an
    # existing segmented/grown tree, else the default fractal tree
    base = (polys and runner.network_from_strokes(polys)) or existing or default

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
    if req.target == "artery":
        project.artery_network = network
    else:
        project.vein_network = network
    store.clear_result(project)   # persist the tree; invalidate stale result
    return GrowInfo(id=project.id, n_segments=network.n_segments,
                    n_terminals=int(network.terminal_nodes.size))


@router.get("/projects/{pid}/network")
def network_geometry(project: Project = Depends(get_project)) -> dict:
    """Segment geometry (domain frame) for overlaying grown/drawn vessels."""
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
