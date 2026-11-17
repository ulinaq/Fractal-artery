"""FastAPI back-end for the perfusion platform (Phase 1 skeleton).

Exposes the compute core as a hosted HTTP service: projects, async simulation
jobs, result retrieval and export. See ``docs/ROADMAP.md`` for the full plan.

Requires the ``api`` extra:  ``uv sync --extra api``.
Run the dev server:  ``uv run perfusion-api``  (or ``uvicorn perfusion.api.app:app``).
"""

from __future__ import annotations


def create_app(*args, **kwargs):
    """Lazily build the FastAPI app (imports FastAPI only when called)."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
