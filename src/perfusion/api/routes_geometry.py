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
from ..core.geometry import rasterize_mask
from . import helpers, runner
from .deps import get_project, get_store, project_info
from .schemas import (
    AdaptInfo,
    AdaptRequest,
    GrowInfo,
    GrowRequest,
    HotspotsIn,
    IOIn,
    LesionsIn,
    MaskIn,
    NetworkModel,
    NetworksEditIn,
    ProjectInfo,
    PruneInfo,
    PruneRequest,
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


@router.get("/projects/{pid}/lesions")
def get_lesions(project: Project = Depends(get_project)) -> dict:
    return {"lesions": project.lesions}


@router.put("/projects/{pid}/lesions", response_model=ProjectInfo)
def set_lesions(body: LesionsIn, project: Project = Depends(get_project),
                store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Set the low-permeability tissue patches (heterogeneous ``K``). Invalidates any
    stale result so the next solve uses the new permeability field."""
    project.lesions = [le.model_dump() for le in body.lesions]
    if project.has_result:
        store.clear_result(project)
    store.save(project)
    return project_info(project)


@router.get("/projects/{pid}/hotspots")
def get_hotspots(project: Project = Depends(get_project)) -> dict:
    return {"hotspots": project.hotspots}


@router.put("/projects/{pid}/hotspots", response_model=ProjectInfo)
def set_hotspots(body: HotspotsIn, project: Project = Depends(get_project),
                 store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Set the data-constrained growth targets: subsequent grows sample attraction
    points with probability weighted by these hotspots (denser tree where demand is
    higher)."""
    project.hotspots = [h.model_dump() for h in body.hotspots]
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
    topology rebuilds it. Sending topology also lets the editor *create* a tree
    the project doesn't have yet (drawn from scratch on the canvas) and *remove*
    one (zero segments). Any change invalidates the result (geometry changed).
    """
    def apply(net, edit):
        topology = edit.seg_start is not None and edit.seg_end is not None
        if topology and len(edit.seg_start) == 0:
            return None                       # every edge deleted -> no tree
        if net is None and not topology:
            raise HTTPException(404, "project has no editable arterial/venous tree")
        if net is not None and net.is_3d:
            raise HTTPException(409, "a 3-D tree cannot be edited in the 2-D editor")
        if topology:
            # topology may have changed (add/delete/new tree) -> rebuild from scratch
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

    if body.artery is None and body.vein is None:
        raise HTTPException(404, "project has no editable arterial/venous tree")
    if body.artery is not None:
        project.artery_network = apply(project.artery_network, body.artery)
    if body.vein is not None:
        project.vein_network = apply(project.vein_network, body.vein)
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
    req = body or GrowRequest()
    if req.method not in ("colonization", "cco"):
        raise HTTPException(422, "method must be 'colonization' or 'cco'")
    if req.target not in ("artery", "vein"):
        raise HTTPException(422, "target must be 'artery' or 'vein'")
    domain_limit = project.grid.domain_limit
    nx, ny = project.grid.nx, project.grid.ny
    mask = (rasterize_mask(nx, ny, domain_limit, project.regions)
            if project.regions else np.ones((nx, ny), bool))
    # data-constrained growth: a demand map from the drawn hotspots (baseline 1)
    density = None
    if project.hotspots:
        from ..core.geometry import rasterize_permeability
        density = rasterize_permeability(
            nx, ny, domain_limit,
            [(h["polygon"], h["weight"]) for h in project.hotspots], base=1.0)

    d_art, d_ven = runner.default_trees(project)       # fitted into this project's domain
    if req.target == "artery":
        polys, existing, default = project.arteries, project.artery_network, d_art
    else:
        polys, existing, default = project.veins, project.vein_network, d_ven
    # prefer the drawn strokes (bends kept) so growth extends them, else an
    # existing segmented/grown tree, else the default fractal tree
    base = (polys and runner.network_from_strokes(polys)) or existing or default

    if req.method == "cco":
        network = growth.cco_grow_in_mask(
            base, mask, domain_limit, n_terminals=req.n_points, seed=req.seed,
            d_min=req.d_min, leaf_radius=req.leaf_radius, density=density)
    else:
        seed_nodes = None
        if req.from_segments:
            seed_nodes = [int(base.seg_end[s]) for s in req.from_segments
                          if 0 <= s < base.n_segments]
        network = growth.grow_in_mask(
            base, mask, domain_limit,
            n_points=req.n_points, seed=req.seed, seed_nodes=seed_nodes, density=density,
            step=req.step, influence_radius=req.influence_radius,
            kill_radius=req.kill_radius, leaf_radius=req.leaf_radius)
    if req.target == "artery":
        project.artery_network = network
    else:
        project.vein_network = network
    store.clear_result(project)   # persist the tree; invalidate stale result
    return GrowInfo(id=project.id, n_segments=network.n_segments,
                    n_terminals=int(network.terminal_nodes.size))


@router.post("/projects/{pid}/prune", response_model=PruneInfo)
def prune_vessels(body: PruneRequest | None = None, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)) -> PruneInfo:
    """Prune ``steps`` generations of terminal branches off the arterial *or* venous
    tree — the inverse of :func:`grow_vessels`. Works on the same base (drawn strokes,
    else the stored tree, else the default fractal tree) and stores the result."""
    from ..core.geometry import prune_terminal_branches

    req = body or PruneRequest()
    if req.target not in ("artery", "vein"):
        raise HTTPException(422, "target must be 'artery' or 'vein'")
    d_art, d_ven = runner.default_trees(project)       # fitted into this project's domain
    if req.target == "artery":
        polys, existing, default = project.arteries, project.artery_network, d_art
    else:
        polys, existing, default = project.veins, project.vein_network, d_ven
    base = (polys and runner.network_from_strokes(polys)) or existing or default
    network = prune_terminal_branches(base, req.steps)
    if req.target == "artery":
        project.artery_network, project.arteries = network, []
    else:
        project.vein_network, project.veins = network, []
    store.clear_result(project)   # persist the tree; invalidate stale result
    return PruneInfo(id=project.id, n_segments=network.n_segments,
                     n_terminals=int(network.terminal_nodes.size),
                     n_removed=base.n_segments - network.n_segments)


@router.post("/projects/{pid}/adapt", response_model=AdaptInfo)
def adapt_vessels(body: AdaptRequest | None = None, project: Project = Depends(get_project),
                  store: ProjectStore = Depends(get_store)) -> AdaptInfo:
    """Structurally adapt the vessel radii toward Murray's law (Pries–Secomb).

    Wall-shear-driven remodelling evolves the arterial + venous radii until the
    ``|Q| ∝ r^k`` exponent reaches ``target_exponent`` (early-stopping at the closest
    approach). The remodelled trees replace the project's networks; any stale result
    is invalidated."""
    req = body or AdaptRequest()
    try:
        ad = runner.adapt_network(project, target_exponent=req.target_exponent,
                                  rate=req.rate, n_iter=req.n_iter)
    except runner.InputError as e:
        raise HTTPException(400, str(e)) from None
    project.artery_network, project.vein_network = ad.art, ad.ven
    project.arteries, project.veins = [], []      # folded into the networks
    store.clear_result(project)                   # geometry changed → stale result
    return AdaptInfo(id=project.id, murray_start=ad.murray_start,
                     murray_final=ad.murray_exponent, n_iter=ad.n_iter,
                     moved_toward_target=abs(ad.murray_exponent - req.target_exponent)
                     < abs(ad.murray_start - req.target_exponent))


@router.get("/projects/{pid}/morphometry")
def morphometry(target: str = "artery", project: Project = Depends(get_project)) -> dict:
    """Horton–Strahler morphometry of the project's ``target`` tree (artery/vein):
    per-order segment counts + mean length/radius, and the bifurcation / length /
    diameter ratios. Falls back to the default fractal tree when none is drawn."""
    from ..core.morphometry import strahler_summary

    art, ven = runner.build_networks(project, use_vessels=True, defaults=True)
    net = ven if target == "vein" else art
    return {"target": target, **strahler_summary(net)}


@router.get("/projects/{pid}/network")
def network_geometry(project: Project = Depends(get_project)) -> dict:
    """Segment geometry (domain frame) for overlaying grown/drawn vessels.

    A tree the project doesn't have is returned as the domain-fitted **default**
    fractal tree that a run would use, flagged ``"default": true`` per segment, and
    listed in ``defaults`` — so the UI can show what will actually be solved."""
    segments: list[dict] = []
    defaults: list[str] = []

    def _add(net, kind: str, default: bool = False) -> None:
        for x0, y0, x1, y1, r in helpers.segment_lines(net):
            seg = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "r": r, "kind": kind}
            if default:
                seg["default"] = True
            segments.append(seg)

    if project.artery_network is not None:
        _add(project.artery_network, "artery")
    elif project.arteries:
        _add(runner.polylines_to_network(project.arteries), "artery")
    if project.vein_network is not None:
        _add(project.vein_network, "vein")
    elif project.veins:
        _add(runner.polylines_to_network(project.veins), "vein")
    has_art = project.artery_network is not None or bool(project.arteries)
    has_ven = project.vein_network is not None or bool(project.veins)
    if not (has_art and has_ven):
        d_art, d_ven = runner.default_trees(project)
        if not has_art:
            _add(d_art, "artery", default=True)
            defaults.append("artery")
        if not has_ven:
            _add(d_ven, "vein", default=True)
            defaults.append("vein")
    return {"segments": segments, "defaults": defaults}
