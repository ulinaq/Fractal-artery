"""Tests for the optional RQ (Redis Queue) job backend.

Uses ``fakeredis`` + a synchronous RQ queue (``is_async=False``) so the full
enqueue -> execute -> persist path runs in-process without a real Redis server or
worker. Needs the ``rq`` extra (+ ``fakeredis`` dev dep).
"""

import time

import pytest

pytest.importorskip("rq")
pytest.importorskip("fakeredis")

from fakeredis import FakeStrictRedis
from rq import Queue

from perfusion.api.jobs import RQJobManager
from perfusion.api.schemas import GridIn, ParametersIn
from perfusion.api.store import ProjectStore

SMALL_GRID = {"nx": 50, "ny": 60, "lx": 100, "ly": 120, "domain_limit": [0, 100, 0, 120]}


def _sync_queue():
    return Queue("test", connection=FakeStrictRedis(), is_async=False)


def test_rq_manager_runs_and_persists_a_job(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create("rq", ParametersIn(r_min_frac=0.5, seed=0), GridIn(**SMALL_GRID))
    jobs = RQJobManager(store, _sync_queue())

    jid = jobs.submit(project.id, "run", {"nonlinear": False, "use_vessels": False})
    # is_async=False executes inline, so by now the job has run
    state = jobs.get(jid)
    assert state["state"] == "done", state
    assert state["kind"] == "run"

    # the result was persisted by the worker path (run_job -> execute_job -> runner)
    reopened = store.get(project.id)
    assert reopened.has_result
    assert reopened.summary["arterial_inflow"] > 0


def test_rq_job_records_errors(tmp_path):
    store = ProjectStore(tmp_path)
    jobs = RQJobManager(store, _sync_queue())
    # a non-existent project id makes the worker raise -> job state 'error'
    jid = jobs.submit("does-not-exist", "run", {})
    deadline = time.time() + 5
    while jobs.get(jid)["state"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.02)
    assert jobs.get(jid)["state"] == "error"
