"""FastAPI application: wires the compute core into a hosted service.

Thin now — the endpoints live in the ``routes_*`` modules (grouped by concern)
and reach the store/job-manager through ``deps`` (held on ``app.state``). This
module only builds the app, picks the job backend, exposes the two app-level
meta endpoints, and mounts the static frontend.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .jobs import JobManager, RQJobManager
from .store import ProjectStore


def _default_jobs(store: ProjectStore):
    """Pick the job backend: RQ if ``PERFUSION_QUEUE=rq`` (and reachable), else threads."""
    import os

    if os.environ.get("PERFUSION_QUEUE", "").lower() == "rq":
        try:
            from redis import Redis
            from rq import Queue

            conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
            conn.ping()
            return RQJobManager(store, Queue("perfusion", connection=conn))
        except Exception:            # rq/redis missing or unreachable -> threads
            pass
    return JobManager(store)


def create_app(store: ProjectStore | None = None,
               jobs: JobManager | None = None) -> FastAPI:
    """Build the FastAPI app; inject a store/job-manager for testing."""
    from .. import __version__
    from . import routes_explore, routes_geometry, routes_projects, routes_solve

    store = store or ProjectStore()
    jobs = jobs or _default_jobs(store)
    app = FastAPI(title="perfusion API", version=__version__,
                  summary="Multiscale blood-flow & perfusion modelling service")
    app.state.store = store       # reached by the route deps (get_store / get_jobs)
    app.state.jobs = jobs

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "build": _frontend_build()}

    @app.get("/frontend-version")
    def frontend_version() -> dict:
        """Current frontend build hash — the open page polls this to detect a
        stale tab (files changed on disk since it loaded) and offer a reload."""
        return {"build": _frontend_build()}

    app.include_router(routes_projects.router)
    app.include_router(routes_geometry.router)
    app.include_router(routes_solve.router)
    app.include_router(routes_explore.router)
    _mount_frontend(app)
    return app


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "frontend"


def _frontend_build() -> str:
    """A short hash of the frontend files' (name, size, mtime).

    Changes whenever any .html/.js/.css asset is edited, so an open tab can tell
    it is running a stale build and offer a reload. Cheap: a handful of files.
    """
    import hashlib

    frontend = _frontend_dir()
    if not frontend.exists():
        return "none"
    parts = []
    for f in sorted(frontend.glob("*")):
        if f.suffix in (".html", ".js", ".css") and f.is_file():
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


def _mount_frontend(app: FastAPI) -> None:
    """Serve the static 2-D workspace at ``/`` if it is present (dev/editable)."""
    frontend = _frontend_dir()
    if not (frontend / "index.html").exists():
        return
    from fastapi.staticfiles import StaticFiles

    class _NoCacheStatic(StaticFiles):
        # Force the browser to revalidate (ETag) every load, so an updated app.js/
        # index.html is never served stale from cache.
        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    # Mounted last, so the API routes above take precedence over "/".
    app.mount("/", _NoCacheStatic(directory=frontend, html=True), name="frontend")


app = create_app()


def main() -> None:
    """Run the dev server (``perfusion-api``). Port defaults to 8010 (8000 is a
    common conflict); override with ``PERFUSION_PORT``."""
    import os

    import uvicorn

    from .store import _default_root
    port = int(os.environ.get("PERFUSION_PORT", "8010"))
    print(f"perfusion-api: http://127.0.0.1:{port}  (set PERFUSION_PORT to change)")
    print(f"perfusion-api: persisting projects under {_default_root()}  "
          f"(set PERFUSION_DATA to change)")
    uvicorn.run("perfusion.api.app:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
