"""Shared FastAPI dependencies + response helpers for the route modules.

The store and job-manager are held on ``app.state`` (set in ``create_app``) and
reached through these dependencies, so routers need no closure over them.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from .schemas import ProjectInfo
from .store import Project, ProjectStore


def get_store(request: Request) -> ProjectStore:
    return request.app.state.store


def get_jobs(request: Request):
    return request.app.state.jobs


def get_project(pid: str, store: ProjectStore = Depends(get_store)) -> Project:
    """Resolve ``{pid}`` to a :class:`Project`, 404-ing if it does not exist."""
    try:
        return store.get(pid)
    except KeyError:
        raise HTTPException(404, f"project {pid!r} not found") from None


def project_info(project: Project) -> ProjectInfo:
    """The public summary of a project (also reflects its params/grid to the UI)."""
    has_mask = bool(project.regions) or getattr(project, "mask", None) is not None
    return ProjectInfo(id=project.id, name=project.name,
                       has_vessels=project.has_vessels, has_mask=has_mask,
                       has_result=project.has_result, has_tracer=project.has_tracer,
                       params=project.params, grid=project.grid)
