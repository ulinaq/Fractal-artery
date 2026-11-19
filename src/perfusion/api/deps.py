"""Shared FastAPI dependencies + response helpers for the route modules.

The store and job-manager are held on ``app.state`` (set in ``create_app``) and
reached through these dependencies, so routers need no closure over them.
"""

from __future__ import annotations

from contextlib import contextmanager

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


def get_project_2d(request: Request, project: Project = Depends(get_project)) -> Project:
    """Like :func:`get_project`, for 2-D-only features: a clear 409 on a 3-D project."""
    if project.grid.is_3d:
        raise HTTPException(409, f"{request.url.path.rsplit('/', 1)[-1]} works on 2-D "
                                 "projects; this is a 3-D project — view and run it in "
                                 "the 3-D viewer (/viewer3d.html?project=…).")
    return project


@contextmanager
def solve_errors():
    """Turn a synchronous re-solve's failures into 4xx instead of a 500: an
    ``InputError`` (the model cannot be built — empty / split mask, no terminals, …)
    is a 400, any other ``ValueError`` (bad option values) a 422."""
    from .runner import InputError

    try:
        yield
    except InputError as e:
        raise HTTPException(400, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


def check_region_dims(project, regions, what: str) -> None:
    """A 3-D project takes spheres ([x, y, z] + radius); a 2-D one polygons or discs."""
    want = 3 if project.grid.is_3d else 2
    for i, r in enumerate(regions):
        if r.dim != want:
            shape = ("a sphere: center [x, y, z] + radius" if want == 3
                     else "a polygon or a disc: center [x, y] + radius")
            raise HTTPException(422, f"{what} {i}: a {want}-D project needs {shape}")


def check_document(doc) -> None:
    """Reject a project document whose parts disagree with its grid (422): lesion /
    hotspot shapes of the wrong dimension, a mask of the wrong shape, or trees that
    are 3-D on a 2-D grid (or vice versa)."""
    from types import SimpleNamespace

    holder = SimpleNamespace(grid=doc.grid)
    check_region_dims(holder, doc.lesions, "lesion")
    check_region_dims(holder, doc.hotspots, "hotspot")
    g = doc.grid
    want = (g.nx, g.ny, g.nz) if g.is_3d else (g.nx, g.ny)
    if doc.mask is not None and tuple(doc.mask.shape) != want:
        raise HTTPException(422, f"mask shape {tuple(doc.mask.shape)} does not match the "
                                 f"grid {want}")
    for name, net in (("artery_network", doc.artery_network),
                      ("vein_network", doc.vein_network)):
        if net is not None and (net.node_z is not None) != g.is_3d:
            raise HTTPException(422, f"{name} is {'3' if net.node_z is not None else '2'}-D "
                                     f"but the grid is {'3' if g.is_3d else '2'}-D")
