"""Async job runners backed by the durable job registry in the store.

Two interchangeable backends behind one ``submit(project_id, kind, req_data)`` /
``get(jid)`` interface:

- :class:`JobManager` — an in-process thread pool (the default; zero infra).
- :class:`RQJobManager` — an RQ (Redis Queue) backend for real multi-worker
  hosting (optional ``rq`` extra + a running Redis and ``rq worker``).

Both record each job's lifecycle (pending -> running -> done/error) in the
:class:`~perfusion.api.store.ProjectStore` so status/history survive a restart
(stale in-flight jobs are reaped to ``interrupted`` on startup). The actual work
lives in :mod:`perfusion.api.runner`, and :func:`run_job` reconstructs the store
from its root path so an RQ worker in another process can run it.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from . import runner
from .schemas import RunRequest, TracerRequest


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def execute_job(store, jid: str, project_id: str, kind: str, req_data: dict) -> None:
    """Run one job against ``store`` and record its lifecycle in the registry."""
    store.set_job_state(jid, "running")
    try:
        project = store.get(project_id)
        if kind == "tracer":
            runner.run_tracer(store, project, TracerRequest(**req_data))
        else:
            runner.run_project(store, project, RunRequest(**req_data))
    except Exception as exc:  # report any failure to the client via job state
        store.set_job_state(jid, "error", _describe_error(exc))
    else:
        store.set_job_state(jid, "done")


def _describe_error(exc: Exception) -> str:
    """A useful error string for the UI: a clean message for known input problems,
    otherwise the exception type + message + a short traceback tail for debugging."""
    from .runner import InputError

    if isinstance(exc, InputError):
        return str(exc)                       # already user-facing and actionable
    import traceback

    head = f"{type(exc).__name__}: {exc}".strip()
    tail = traceback.format_exc().strip().splitlines()[-3:]
    return head + "\n\nwhere:\n" + "\n".join(tail)


def run_job(store_root: str, jid: str, project_id: str, kind: str, req_data: dict) -> None:
    """Top-level, picklable entry point for an RQ worker (reopens the store)."""
    from .store import ProjectStore

    execute_job(ProjectStore(store_root), jid, project_id, kind, req_data)


class JobManager:
    """In-process thread-pool backend (default)."""

    def __init__(self, store, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._store = store

    def submit(self, project_id: str, kind: str, req_data: dict) -> str:
        jid = _new_id()
        self._store.create_job(jid, project_id, kind, state="pending")
        self._executor.submit(execute_job, self._store, jid, project_id, kind, req_data)
        return jid

    def get(self, jid: str) -> dict:
        job = self._store.get_job(jid)
        if job is None:
            raise KeyError(jid)
        return job


class RQJobManager:
    """RQ (Redis Queue) backend: enqueue jobs for separate worker processes."""

    def __init__(self, store, queue) -> None:
        self._store = store
        self._queue = queue

    def submit(self, project_id: str, kind: str, req_data: dict) -> str:
        jid = _new_id()
        self._store.create_job(jid, project_id, kind, state="pending")
        self._queue.enqueue(run_job, str(self._store.root), jid, project_id, kind,
                            req_data, job_id=jid)
        return jid

    def get(self, jid: str) -> dict:
        job = self._store.get_job(jid)
        if job is None:
            raise KeyError(jid)
        return job
