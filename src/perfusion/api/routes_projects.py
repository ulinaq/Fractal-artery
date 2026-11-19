"""Project lifecycle routes: create / list / read, the frog example, and the
portable save-file document round-trip."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from . import helpers
from .deps import check_document, get_project, get_store, project_info
from .schemas import GridIn, ParametersIn, ProjectCreate, ProjectDocument, ProjectInfo
from .store import (
    Project,
    ProjectStore,
    document_from_project,
    project_from_document,
)

router = APIRouter()


@router.post("/projects", response_model=ProjectInfo)
def create_project(body: ProjectCreate, store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    return project_info(store.create(body.name, body.params, body.grid))


@router.get("/projects", response_model=list[ProjectInfo])
def list_projects(store: ProjectStore = Depends(get_store)) -> list[ProjectInfo]:
    return [project_info(p) for p in store.list()]


@router.get("/projects/{pid}/document")
def export_document(project: Project = Depends(get_project)):
    """Download the whole project as one portable JSON file (save your work)."""
    doc = document_from_project(project)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name) or "project"
    return JSONResponse(
        doc.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="{safe}.perfusion.json"'},
    )


@router.post("/projects/import", response_model=ProjectInfo)
def import_document(doc: ProjectDocument, store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """Restore a saved project document as a *new* project (continue later)."""
    check_document(doc)
    project = project_from_document(doc, uuid.uuid4().hex[:12])
    store.save(project)
    return project_info(project)


@router.post("/projects/frog", response_model=ProjectInfo)
def create_frog(name: str = "frog tongue",
                store: ProjectStore = Depends(get_store)) -> ProjectInfo:
    """One-click example: the segmented frog-tongue vasculature + organ mask.

    Segments the bundled frog artery/vein images and tongue mask and places
    them in the app's standard domain (reusing ``segmentation.network_in_domain``
    / ``mask_in_domain``) — a real coupled system ready to Run. Needs the
    ``image`` extra.
    """
    dom = (0.0, 100.0, 0.0, 120.0)
    nx, ny = 120, 140
    try:
        art, ven, mask = helpers.frog_networks_and_mask(nx, ny, dom)
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
    return project_info(project)


@router.get("/projects/{pid}", response_model=ProjectInfo)
def read_project(project: Project = Depends(get_project)) -> ProjectInfo:
    return project_info(project)
