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

    def to_core(self) -> Grid:
        return Grid(nx=self.nx, ny=self.ny, lx=self.lx, ly=self.ly)


class Polyline(BaseModel):
    """A hand-drawn vessel: an ordered list of ``[x, y]`` points and a radius."""

    points: list[tuple[float, float]]
    radius: float = 1.0


class VesselsIn(BaseModel):
    arteries: list[Polyline] = Field(default_factory=list)
    veins: list[Polyline] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str
    params: ParametersIn = Field(default_factory=ParametersIn)
    grid: GridIn = Field(default_factory=GridIn)


class ProjectInfo(BaseModel):
    id: str
    name: str
    has_vessels: bool = False
    has_result: bool = False


class RunRequest(BaseModel):
    nonlinear: bool = False
    use_vessels: bool = False


class GrowRequest(BaseModel):
    """Grow the arterial tree into the tissue by space colonisation."""

    n_points: int = Field(400, ge=1, le=20000)   # attraction points sampled in the mask
    seed: int = 0
    step: float | None = None                    # None -> auto (fraction of extent)
    influence_radius: float | None = None
    kill_radius: float | None = None
    leaf_radius: float | None = None
    from_segments: list[int] | None = None       # grow only from these segment tips


class GrowInfo(BaseModel):
    id: str
    n_segments: int
    n_terminals: int


class JobInfo(BaseModel):
    id: str
    state: str                 # pending | running | done | error
    project_id: str
    error: str | None = None


class FieldResponse(BaseModel):
    name: str
    shape: tuple[int, int]
    data: list[list[float | None]]   # NaN (masked) cells serialise as null
