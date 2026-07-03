"""In-process async job runner.

Runs simulations on a thread pool and tracks their state. This is the simplest
thing that models the async contract; for real multi-user hosting swap it for a
task queue (RQ/Celery + Redis) behind the same ``submit``/``get`` interface (see
``docs/ROADMAP.md``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict] = {}
        self._lock = Lock()

    def submit(self, func: Callable[[], None], project_id: str) -> str:
        """Schedule ``func`` on a worker thread; return the new job id."""
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = {"state": "pending", "error": None,
                               "project_id": project_id}

        def run() -> None:
            with self._lock:
                self._jobs[jid]["state"] = "running"
            try:
                func()
            except Exception as exc:  # report any failure to the client via job state
                with self._lock:
                    self._jobs[jid].update(state="error", error=str(exc))
            else:
                with self._lock:
                    self._jobs[jid]["state"] = "done"

        self._executor.submit(run)
        return jid

    def get(self, jid: str) -> dict:
        with self._lock:
            return dict(self._jobs[jid])
