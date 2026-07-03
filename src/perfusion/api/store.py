"""In-memory project store (swap for a DB + blob store when scaling — see roadmap)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .schemas import GridIn, ParametersIn, Polyline


@dataclass
class Project:
    id: str
    name: str
    params: ParametersIn
    grid: GridIn
    arteries: list[Polyline] = field(default_factory=list)
    veins: list[Polyline] = field(default_factory=list)
    artery_network: object | None = None  # grown perfusion.core.geometry.Network
    result: object | None = None          # perfusion.core.pipeline.HybridResult
    summary: dict | None = None

    @property
    def has_vessels(self) -> bool:
        return bool(self.arteries or self.veins)

    @property
    def has_result(self) -> bool:
        return self.result is not None


class ProjectStore:
    """Thread-safe-enough (GIL) registry of projects keyed by id."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}

    def create(self, name: str, params: ParametersIn, grid: GridIn) -> Project:
        pid = uuid.uuid4().hex[:12]
        project = Project(id=pid, name=name, params=params, grid=grid)
        self._projects[pid] = project
        return project

    def get(self, pid: str) -> Project:
        return self._projects[pid]

    def list(self) -> list[Project]:
        return list(self._projects.values())
