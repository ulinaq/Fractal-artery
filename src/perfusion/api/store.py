"""Persistent project store: SQLite metadata + a filesystem blob store.

Projects (parameters, grid, hand-drawn vessels, summary) live in a SQLite table;
the heavy binary artefacts — the grown arterial network and the result fields —
live as ``.npz`` blobs beside it. The store therefore survives a process restart,
so a shared ``?project=<id>`` URL keeps working.

The interface (``create`` / ``get`` / ``list`` / ``save`` / ``save_result`` / …)
is deliberately small so it can be swapped for Postgres + object storage when
scaling (see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

from .. import io
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
    summary: dict | None = None
    has_result: bool = False              # set from blob existence on load

    @property
    def has_vessels(self) -> bool:
        return bool(self.arteries or self.veins)


def _default_root() -> Path:
    """Where to persist when the caller does not choose (env-overridable)."""
    env = os.environ.get("PERFUSION_DATA")
    return Path(env) if env else Path(tempfile.gettempdir()) / "perfusion-data"


class ProjectStore:
    """SQLite + blob-backed project registry."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "projects.db"
        self._lock = Lock()
        with self._txn() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS projects (
                       id TEXT PRIMARY KEY, name TEXT NOT NULL,
                       params TEXT NOT NULL, grid TEXT NOT NULL,
                       arteries TEXT NOT NULL, veins TEXT NOT NULL,
                       summary TEXT )"""
            )

    # -- connections / blob paths ------------------------------------------- #
    @contextmanager
    def _txn(self):
        """A short-lived connection that commits on success and always closes."""
        con = sqlite3.connect(self.db_path, timeout=10.0)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def _artery_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.artery.npz"

    def result_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.result.npz"

    # -- (de)serialisation of a row ----------------------------------------- #
    def _row_to_project(self, row: sqlite3.Row) -> Project:
        pid = row["id"]
        project = Project(
            id=pid, name=row["name"],
            params=ParametersIn(**json.loads(row["params"])),
            grid=GridIn(**json.loads(row["grid"])),
            arteries=[Polyline(**d) for d in json.loads(row["arteries"])],
            veins=[Polyline(**d) for d in json.loads(row["veins"])],
            summary=json.loads(row["summary"]) if row["summary"] else None,
        )
        artery = self._artery_path(pid)
        if artery.exists():
            project.artery_network = io.load_network(artery)
        project.has_result = self.result_path(pid).exists()
        return project

    # -- CRUD --------------------------------------------------------------- #
    def create(self, name: str, params: ParametersIn, grid: GridIn) -> Project:
        project = Project(id=uuid.uuid4().hex[:12], name=name, params=params, grid=grid)
        self.save(project)
        return project

    def get(self, pid: str) -> Project:
        with self._txn() as con:
            row = con.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise KeyError(pid)
        return self._row_to_project(row)

    def list(self) -> list[Project]:
        with self._txn() as con:
            rows = con.execute("SELECT * FROM projects ORDER BY rowid").fetchall()
        return [self._row_to_project(r) for r in rows]

    def save(self, project: Project) -> None:
        """Persist metadata + vessels, and sync the artery-network blob."""
        with self._lock, self._txn() as con:
            con.execute(
                """INSERT INTO projects (id, name, params, grid, arteries, veins, summary)
                       VALUES (:id, :name, :params, :grid, :arteries, :veins, :summary)
                   ON CONFLICT(id) DO UPDATE SET
                       name=:name, params=:params, grid=:grid,
                       arteries=:arteries, veins=:veins, summary=:summary""",
                {
                    "id": project.id, "name": project.name,
                    "params": project.params.model_dump_json(),
                    "grid": project.grid.model_dump_json(),
                    "arteries": json.dumps([p.model_dump() for p in project.arteries]),
                    "veins": json.dumps([p.model_dump() for p in project.veins]),
                    "summary": json.dumps(project.summary) if project.summary else None,
                },
            )
        artery = self._artery_path(project.id)
        if project.artery_network is not None:
            io.save_network(project.artery_network, artery)
        else:
            artery.unlink(missing_ok=True)

    def save_result(self, project: Project, result) -> None:
        """Write the result blob + summary and persist."""
        io.save_results(result, self.result_path(project.id))
        project.summary = io.results_summary(result)
        project.has_result = True
        self.save(project)

    def clear_result(self, project: Project) -> None:
        """Invalidate a stale result (e.g. after regrowing the tree)."""
        self.result_path(project.id).unlink(missing_ok=True)
        project.summary = None
        project.has_result = False
        self.save(project)

    def result_field(self, pid: str, name: str) -> np.ndarray:
        """Load one field (``pressure`` / ``flux`` / ``tof``) from the result blob."""
        with np.load(self.result_path(pid)) as data:
            return data[name]
