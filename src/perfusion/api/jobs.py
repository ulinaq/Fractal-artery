"""Async job runner backed by the durable job registry in the store.

Runs simulations on a thread pool and records each job's lifecycle
(pending -> running -> done/error) in the :class:`~perfusion.api.store.ProjectStore`
so status and history survive a process restart (stale in-flight jobs are marked
``interrupted`` on startup). For real multi-user hosting swap the executor for a
task queue (RQ/Celery + Redis) behind the same ``submit``/``get`` interface, and
have the workers write job state through the same store (see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class JobManager:
    def __init__(self, store, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._store = store

    def submit(self, func: Callable[[], None], project_id: str, kind: str = "run") -> str:
        """Schedule ``func`` on a worker thread; return the new job id."""
        jid = uuid.uuid4().hex[:12]
        self._store.create_job(jid, project_id, kind, state="pending")

        def run() -> None:
            self._store.set_job_state(jid, "running")
            try:
                func()
            except Exception as exc:  # report any failure to the client via job state
                self._store.set_job_state(jid, "error", str(exc))
            else:
                self._store.set_job_state(jid, "done")

        self._executor.submit(run)
        return jid

    def get(self, jid: str) -> dict:
        job = self._store.get_job(jid)
        if job is None:
            raise KeyError(jid)
        return job
