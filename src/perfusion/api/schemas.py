"""Pydantic request/response models for the perfusion API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.darcy import Grid
from ..core.parameters import Parameters


class ParametersIn(BaseModel):
    """Flow/geometry parameters (subset of :class:`perfusion.core.Parameters`)."""

    p_in: float = 10.0
    p_out: float = 0.0
    mu: float = 1.0
    hematocrit: float = 0.45
    r0: float = 2.0
    length_to_radius: float = 10.0
    exponent: float = 2.8
    r_min_frac: float = 0.15
    rv_scale: float = 1.2
    seed: int | None = 0
    fahraeus_lindqvist: bool = False
    radius_to_um: float = 1.0

    def to_core(self) -> Parameters:
        return Parameters(**self.model_dump())


class GridIn(BaseModel):
    """Darcy capillary-bed grid and the coordinate window it maps to."""

    nx: int = 140
    ny: int = 160
    lx: float = 140.0
    ly: float = 160.0
    domain_limit: tuple[float, float, float, float] = (-70.0, 70.0, 0.0, 160.0)

    def to_core(self, mask=None) -> Grid:
        return Grid(nx=self.nx, ny=self.ny, lx=self.lx, ly=self.ly, mask=mask)


class Polyline(BaseModel):
    """A hand-drawn vessel: an ordered list of ``[x, y]`` points and a radius."""

    points: list[tuple[float, float]]
    radius: float = 1.0


class VesselsIn(BaseModel):
    arteries: list[Polyline] = Field(default_factory=list)
    veins: list[Polyline] = Field(default_factory=list)


class MaskIn(BaseModel):
    """Tissue domain as a union of hand-drawn outline polygons (domain coords)."""

    regions: list[list[tuple[float, float]]] = Field(default_factory=list)


class IOIn(BaseModel):
    """Inlet (arterial, p_in) and outlet (venous, p_out) points, in domain coords.

    Each point is snapped to the nearest vessel end; list several for multiple
    inlets/outlets. Empty falls back to the single root inlet/outlet.
    """

    inlets: list[tuple[float, float]] = Field(default_factory=list)
    outlets: list[tuple[float, float]] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str
    params: ParametersIn = Field(default_factory=ParametersIn)
    grid: GridIn = Field(default_factory=GridIn)


class ProjectInfo(BaseModel):
    id: str
    name: str
    has_vessels: bool = False
    has_mask: bool = False
    has_result: bool = False
    has_tracer: bool = False
    params: ParametersIn | None = None     # so the UI can reflect a project's settings
    grid: GridIn | None = None


class RunRequest(BaseModel):
    model: str = "hybrid"                          # "hybrid" (Darcy bed) | "direct" (connections)
    nonlinear: bool = False
    use_vessels: bool = False
    max_iter: int = Field(100, ge=1, le=100000)   # nonlinear fixed-point iteration cap
    tol: float = Field(1e-10, gt=0)               # convergence tolerance
    rho_c: float = Field(0.0, ge=0)               # direct model: terminal connection radius
    phase_separation: bool = False                 # RBC plasma-skimming (implies Fåhræus–Lindqvist)
    elasticity: bool = False                       # pressure-dependent vessel radii
    compliance: float = Field(0.01, ge=0)          # radius change per unit transmural pressure


class ExploreRequest(BaseModel):
    """A project-independent fractal-tree evaluation for the Tree Explorer.

    Builds an ``optimize.alpha`` vector from these tunables, solves the hybrid
    model on a coarse grid, and returns a field + the tree geometry + the chosen
    homogeneity objective.
    """

    a_gamma: float = Field(1.4, ge=1.0, le=1.9)          # arterial branching asymmetry
    v_gamma: float = Field(1.2, ge=1.0, le=1.9)          # venous branching asymmetry
    length_to_radius: float = Field(10.0, ge=4.0, le=18.0)
    exponent: float = Field(2.8, ge=2.0, le=3.5)         # Murray bifurcation exponent
    rv_scale: float = Field(1.2, ge=0.8, le=1.8)         # venous radius scale
    kind: str = "tof_std"                                # objective (flux_*/tof_*)
    field: str = "flux"                                  # field to render (pressure/flux/tof)
    nx: int = Field(60, ge=20, le=140)
    ny: int = Field(70, ge=20, le=160)


class ExploreOptimizeRequest(BaseModel):
    """Optimise the tree for homogeneity (Tree Explorer 'Optimise' button)."""

    kind: str = "tof_std"
    method: str = "local"                                # "local" | "global"
    maxiter: int = Field(15, ge=1, le=60)
    nx: int = Field(50, ge=20, le=120)
    ny: int = Field(58, ge=20, le=140)


class GrowRequest(BaseModel):
    """Grow the arterial tree into the tissue.

    ``method="colonization"`` (default) uses space colonisation with ``n_points``
    attraction points; ``method="cco"`` uses constrained constructive
    optimisation, adding ``n_points`` terminals spaced at least ``d_min`` apart.
    """

    target: str = "artery"                        # "artery" | "vein" — which tree to grow
    method: str = "colonization"                  # "colonization" | "cco"
    n_points: int = Field(400, ge=1, le=20000)   # attraction points / CCO terminals
    seed: int = 0
    step: float | None = None                    # colonisation: None -> auto
    influence_radius: float | None = None
    kill_radius: float | None = None
    d_min: float | None = None                   # cco: min terminal spacing
    leaf_radius: float | None = None
    from_segments: list[int] | None = None       # colonisation: grow from these tips


class GrowInfo(BaseModel):
    id: str
    n_segments: int
    n_terminals: int


class TracerRequest(BaseModel):
    """Animate a contrast-agent bolus washing through the tissue."""

    n_frames: int = Field(48, ge=2, le=200)
    duration: float | None = None   # total time; auto from time-of-flight if None
    aif_t0: float | None = None      # bolus arrival time; auto (~8% of duration)


class TracerInfo(BaseModel):
    id: str
    n_frames: int
    shape: tuple[int, int]
    times: list[float]
    vmax: float
    frames: list[list[list[float | None]]]   # (n_frames, nx, ny); masked -> null


class NetworkModel(BaseModel):
    """A vascular :class:`~perfusion.core.geometry.Network` as plain JSON.

    Carries the full topology + geometry (and, unlike the ``.npz`` blob,
    ``inlet_nodes``/``node_z`` too) so a project round-trips losslessly.
    """

    node_x: list[float]
    node_y: list[float]
    node_role: list[int]
    seg_start: list[int]
    seg_end: list[int]
    seg_length: list[float]
    seg_radius: list[float]
    seg_angle: list[float]
    root_node: int = 0
    inlet_nodes: list[int] | None = None
    node_z: list[float] | None = None

    @classmethod
    def from_core(cls, net) -> NetworkModel:
        import numpy as np

        def arr(x, dtype=float):
            return np.asarray(x, dtype=dtype).tolist()

        return cls(
            node_x=arr(net.node_x), node_y=arr(net.node_y),
            node_role=arr(net.node_role, int),
            seg_start=arr(net.seg_start, int), seg_end=arr(net.seg_end, int),
            seg_length=arr(net.seg_length), seg_radius=arr(net.seg_radius),
            seg_angle=arr(net.seg_angle), root_node=int(net.root_node),
            inlet_nodes=arr(net.inlet_nodes, int) if net.inlet_nodes is not None else None,
            node_z=arr(net.node_z) if net.node_z is not None else None,
        )

    def to_core(self):
        import numpy as np

        from ..core.geometry import Network

        return Network(
            node_x=np.asarray(self.node_x, float), node_y=np.asarray(self.node_y, float),
            node_role=np.asarray(self.node_role, int),
            seg_start=np.asarray(self.seg_start, int), seg_end=np.asarray(self.seg_end, int),
            seg_length=np.asarray(self.seg_length, float),
            seg_radius=np.asarray(self.seg_radius, float),
            seg_angle=np.asarray(self.seg_angle, float), root_node=int(self.root_node),
            inlet_nodes=(np.asarray(self.inlet_nodes, int)
                         if self.inlet_nodes is not None else None),
            node_z=np.asarray(self.node_z, float) if self.node_z is not None else None,
        )


class MaskModel(BaseModel):
    """A raster tissue mask: ``shape`` + a row-major flat list of booleans."""

    shape: tuple[int, int]
    data: list[bool]

    @classmethod
    def from_core(cls, mask) -> MaskModel:
        import numpy as np

        m = np.asarray(mask, dtype=bool)
        return cls(shape=(int(m.shape[0]), int(m.shape[1])), data=m.reshape(-1).tolist())

    def to_core(self):
        import numpy as np

        return np.asarray(self.data, dtype=bool).reshape(self.shape)


class ProjectDocument(BaseModel):
    """A complete, self-contained project — everything needed to resume work later.

    Download one with ``GET /projects/{id}/document`` and restore it (as a new
    project) with ``POST /projects/import``. Everything is plain JSON, so the file
    is portable, diff-able and editable by hand.
    """

    version: int = 1
    name: str = "workspace"
    params: ParametersIn = Field(default_factory=ParametersIn)
    grid: GridIn = Field(default_factory=GridIn)
    arteries: list[Polyline] = Field(default_factory=list)
    veins: list[Polyline] = Field(default_factory=list)
    regions: list[list[tuple[float, float]]] = Field(default_factory=list)
    inlets: list[tuple[float, float]] = Field(default_factory=list)
    outlets: list[tuple[float, float]] = Field(default_factory=list)
    artery_network: NetworkModel | None = None
    vein_network: NetworkModel | None = None
    mask: MaskModel | None = None
    summary: dict | None = None


class NetworkEditIn(BaseModel):
    """Edited geometry (and optionally topology) for one tree.

    ``node_x``/``node_y`` give every node's position and ``seg_radius`` every
    segment's radius. Omit ``seg_start``/``seg_end`` for a pure move/resize
    (topology preserved from the stored tree); include them to also add/delete
    nodes and segments — the server then rebuilds the tree, drops isolated
    (degree-0, non-inlet) nodes, reindexes, and recomputes lengths/angles.
    """

    node_x: list[float]
    node_y: list[float]
    seg_radius: list[float]
    seg_start: list[int] | None = None
    seg_end: list[int] | None = None
    inlet_nodes: list[int] | None = None   # prescribed-pressure nodes (new indices)
    root_node: int | None = None


class NetworksEditIn(BaseModel):
    """Interactive edits to the arterial and/or venous trees."""

    artery: NetworkEditIn | None = None
    vein: NetworkEditIn | None = None


class JobInfo(BaseModel):
    id: str
    state: str                 # pending | running | done | error | interrupted
    project_id: str
    error: str | None = None
    kind: str | None = None    # run | tracer
    created_at: str | None = None
    updated_at: str | None = None


class FieldResponse(BaseModel):
    name: str
    shape: tuple[int, int]
    data: list[list[float | None]]   # NaN (masked) cells serialise as null
