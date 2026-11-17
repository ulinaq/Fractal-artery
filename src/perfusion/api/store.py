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
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np

from .. import io
from .schemas import (
    GridIn,
    MaskModel,
    NetworkModel,
    ParametersIn,
    Polyline,
    ProjectDocument,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Project:
    id: str
    name: str
    params: ParametersIn
    grid: GridIn
    arteries: list[Polyline] = field(default_factory=list)
    veins: list[Polyline] = field(default_factory=list)
    regions: list = field(default_factory=list)  # tissue outline polygons
    lesions: list = field(default_factory=list)  # low-K patches: {polygon, permeability}
    hotspots: list = field(default_factory=list)  # growth-demand targets: {polygon, weight}
    inlets: list = field(default_factory=list)    # arterial inlet points (p_in)
    outlets: list = field(default_factory=list)   # venous outlet points (p_out)
    artery_network: object | None = None  # grown/segmented perfusion.core.geometry.Network
    vein_network: object | None = None    # segmented venous network
    mask: object | None = None            # optional raster tissue mask (nx, ny) bool
    summary: dict | None = None
    has_result: bool = False              # set from blob existence on load
    has_tracer: bool = False              # set from blob existence on load

    @property
    def has_vessels(self) -> bool:
        return bool(self.arteries or self.veins)


def document_from_project(project: Project) -> ProjectDocument:
    """Snapshot a project as a portable, self-contained JSON document."""
    return ProjectDocument(
        name=project.name, params=project.params, grid=project.grid,
        arteries=list(project.arteries), veins=list(project.veins),
        regions=project.regions, lesions=project.lesions, hotspots=project.hotspots,
        inlets=project.inlets, outlets=project.outlets,
        artery_network=(NetworkModel.from_core(project.artery_network)
                        if project.artery_network is not None else None),
        vein_network=(NetworkModel.from_core(project.vein_network)
                      if project.vein_network is not None else None),
        mask=MaskModel.from_core(project.mask) if project.mask is not None else None,
        summary=project.summary,
    )


def project_from_document(doc: ProjectDocument, pid: str) -> Project:
    """Rebuild a :class:`Project` (with a fresh ``pid``) from a saved document."""
    return Project(
        id=pid, name=doc.name, params=doc.params, grid=doc.grid,
        arteries=list(doc.arteries), veins=list(doc.veins),
        regions=[[list(pt) for pt in poly] for poly in doc.regions],
        lesions=[le.model_dump() for le in doc.lesions],
        hotspots=[h.model_dump() for h in doc.hotspots],
        inlets=[list(pt) for pt in doc.inlets], outlets=[list(pt) for pt in doc.outlets],
        artery_network=doc.artery_network.to_core() if doc.artery_network else None,
        vein_network=doc.vein_network.to_core() if doc.vein_network else None,
        mask=doc.mask.to_core() if doc.mask else None,
        summary=doc.summary,
    )


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
                       regions TEXT NOT NULL DEFAULT '[]',
                       lesions TEXT NOT NULL DEFAULT '[]',
                       hotspots TEXT NOT NULL DEFAULT '[]',
                       inlets TEXT NOT NULL DEFAULT '[]',
                       outlets TEXT NOT NULL DEFAULT '[]', summary TEXT )"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                       id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                       kind TEXT NOT NULL, state TEXT NOT NULL, error TEXT,
                       created_at TEXT NOT NULL, updated_at TEXT NOT NULL )"""
            )
            # migrate DBs created before these columns existed
            cols = {r["name"] for r in con.execute("PRAGMA table_info(projects)")}
            for col in ("regions", "lesions", "hotspots", "inlets", "outlets"):
                if col not in cols:
                    con.execute(f"ALTER TABLE projects ADD COLUMN {col} TEXT NOT NULL DEFAULT '[]'")
            # a restart killed any in-flight jobs -> mark them interrupted
            con.execute("UPDATE jobs SET state='interrupted', updated_at=? "
                        "WHERE state IN ('pending','running')", (_now(),))

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

    def _vein_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.vein.npz"

    def _mask_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.rmask.npz"

    def result_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.result.npz"

    def _direct_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.direct.npz"

    def _nettracer_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.nettracer.npz"

    def tracer_path(self, pid: str) -> Path:
        return self.blobs / f"{pid}.tracer.npz"

    # -- (de)serialisation of a row ----------------------------------------- #
    def _row_to_project(self, row: sqlite3.Row) -> Project:
        pid = row["id"]
        project = Project(
            id=pid, name=row["name"],
            params=ParametersIn(**json.loads(row["params"])),
            grid=GridIn(**json.loads(row["grid"])),
            arteries=[Polyline(**d) for d in json.loads(row["arteries"])],
            veins=[Polyline(**d) for d in json.loads(row["veins"])],
            regions=json.loads(row["regions"]) if row["regions"] else [],
            lesions=json.loads(row["lesions"]) if row["lesions"] else [],
            hotspots=json.loads(row["hotspots"]) if row["hotspots"] else [],
            inlets=json.loads(row["inlets"]) if row["inlets"] else [],
            outlets=json.loads(row["outlets"]) if row["outlets"] else [],
            summary=json.loads(row["summary"]) if row["summary"] else None,
        )
        artery = self._artery_path(pid)
        if artery.exists():
            project.artery_network = io.load_network(artery)
        vein = self._vein_path(pid)
        if vein.exists():
            project.vein_network = io.load_network(vein)
        mask = self._mask_path(pid)
        if mask.exists():
            with np.load(mask) as d:
                project.mask = d["mask"]
        project.has_result = self.result_path(pid).exists() or self._direct_path(pid).exists()
        project.has_tracer = self.tracer_path(pid).exists()
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
                """INSERT INTO projects
                       (id, name, params, grid, arteries, veins, regions, lesions,
                        hotspots, inlets, outlets, summary)
                       VALUES (:id, :name, :params, :grid, :arteries, :veins,
                               :regions, :lesions, :hotspots, :inlets, :outlets, :summary)
                   ON CONFLICT(id) DO UPDATE SET
                       name=:name, params=:params, grid=:grid, arteries=:arteries,
                       veins=:veins, regions=:regions, lesions=:lesions, hotspots=:hotspots,
                       inlets=:inlets, outlets=:outlets, summary=:summary""",
                {
                    "id": project.id, "name": project.name,
                    "params": project.params.model_dump_json(),
                    "grid": project.grid.model_dump_json(),
                    "arteries": json.dumps([p.model_dump() for p in project.arteries]),
                    "veins": json.dumps([p.model_dump() for p in project.veins]),
                    "regions": json.dumps(project.regions),
                    "lesions": json.dumps(project.lesions),
                    "hotspots": json.dumps(project.hotspots),
                    "inlets": json.dumps(project.inlets),
                    "outlets": json.dumps(project.outlets),
                    "summary": json.dumps(project.summary) if project.summary else None,
                },
            )
        for path, net in ((self._artery_path(project.id), project.artery_network),
                          (self._vein_path(project.id), project.vein_network)):
            if net is not None:
                io.save_network(net, path)
            else:
                path.unlink(missing_ok=True)
        mask_path = self._mask_path(project.id)
        if project.mask is not None:
            np.savez(mask_path, mask=np.asarray(project.mask, dtype=bool))
        else:
            mask_path.unlink(missing_ok=True)

    def save_result(self, project: Project, result, notes=None) -> None:
        """Write the result blob + summary and persist (invalidating any tracer).

        ``notes`` (e.g. a non-convergence warning) are stored in the summary so the
        UI can surface them.
        """
        io.save_results(result, self.result_path(project.id))
        self.tracer_path(project.id).unlink(missing_ok=True)   # stale
        self._nettracer_path(project.id).unlink(missing_ok=True)
        self._direct_path(project.id).unlink(missing_ok=True)  # switching model → drop direct blob
        summary = io.results_summary(result)
        if notes:
            summary["notes"] = list(notes)
        project.summary = summary
        project.has_result = True
        project.has_tracer = False
        self.save(project)

    def clear_result(self, project: Project) -> None:
        """Invalidate a stale result + tracer (e.g. after regrowing the tree)."""
        self.result_path(project.id).unlink(missing_ok=True)
        self._direct_path(project.id).unlink(missing_ok=True)
        self.tracer_path(project.id).unlink(missing_ok=True)
        self._nettracer_path(project.id).unlink(missing_ok=True)
        project.summary = None
        project.has_result = project.has_tracer = False
        self.save(project)

    def save_direct_result(self, project: Project, art, ven, sol) -> None:
        """Persist a direct-connection result: a summary + a render blob (vessel
        segments and terminal-connection links, with flow magnitudes)."""
        def seg_lines(net, q):
            return np.column_stack([
                net.node_x[net.seg_start], net.node_y[net.seg_start],
                net.node_x[net.seg_end], net.node_y[net.seg_end], np.abs(q),
            ]).astype(float)

        conn = np.array([[art.node_x[c.arterial_terminal], art.node_y[c.arterial_terminal],
                          ven.node_x[c.venous_terminal], ven.node_y[c.venous_terminal], abs(qc)]
                         for c, qc in zip(sol.connections, sol.q_c, strict=True)], dtype=float)
        if conn.size == 0:
            conn = np.zeros((0, 5))
        np.savez(self._direct_path(project.id), artery=seg_lines(art, sol.q_a),
                 vein=seg_lines(ven, sol.q_v), conn=conn)
        self.result_path(project.id).unlink(missing_ok=True)   # not a field result
        self.tracer_path(project.id).unlink(missing_ok=True)
        self._nettracer_path(project.id).unlink(missing_ok=True)   # stale animation
        perfusion = float(np.abs(sol.q_c).sum())
        p_all = np.concatenate([sol.p_a, sol.p_v])
        project.summary = {
            "model": "direct", "arterial_inflow": perfusion, "venous_outflow": perfusion,
            "n_connections": len(sol.connections), "pressure_max": float(p_all.max()),
            "flux_mean": float(conn[:, 4].mean() if len(conn) else 0.0),
        }
        project.has_result = True
        project.has_tracer = False
        self.save(project)

    def load_direct(self, pid: str) -> dict:
        with np.load(self._direct_path(pid)) as d:
            return {k: d[k] for k in d.files}

    def save_network_tracer(self, project: Project, lines, kind, frames, times,
                            arrivals, vmax) -> None:
        """Persist the direct-model network-transit animation (element lines +
        per-element concentration frames); the direct model has no grid field."""
        np.savez(self._nettracer_path(project.id),
                 lines=np.asarray(lines, dtype=float), kind=np.asarray(kind, dtype=np.int8),
                 frames=np.asarray(frames, dtype=np.float32),
                 times=np.asarray(times, dtype=float),
                 arrivals=np.asarray(arrivals, dtype=float), vmax=float(vmax))

    def load_network_tracer(self, pid: str) -> dict:
        with np.load(self._nettracer_path(pid)) as d:
            return {k: d[k] for k in d.files}

    def result_field(self, pid: str, name: str) -> np.ndarray:
        """Load one field (``pressure`` / ``flux`` / ``tof``) from the result blob."""
        with np.load(self.result_path(pid)) as data:
            return data[name]

    def save_tracer(self, project: Project, frames, times, vmax, mask, aif=None) -> None:
        """Persist the tracer animation frames as a blob (with the driving AIF, so
        tracer-kinetic fitting has the exact arterial input)."""
        arrays = dict(frames=np.asarray(frames, dtype=np.float32),
                      times=np.asarray(times, dtype=float), vmax=float(vmax),
                      mask=np.asarray(mask, dtype=bool))
        if aif is not None:
            arrays["aif"] = np.asarray(aif, dtype=float)
        np.savez(self.tracer_path(project.id), **arrays)
        project.has_tracer = True

    def load_tracer(self, pid: str) -> dict:
        """Load the tracer blob (frames, times, vmax, mask) as a dict of arrays."""
        with np.load(self.tracer_path(pid)) as data:
            return {k: data[k] for k in data.files}

    # -- jobs (durable registry) -------------------------------------------- #
    def create_job(self, jid: str, project_id: str, kind: str,
                   state: str = "pending") -> None:
        now = _now()
        with self._lock, self._txn() as con:
            con.execute(
                """INSERT INTO jobs (id, project_id, kind, state, error,
                                     created_at, updated_at)
                       VALUES (?, ?, ?, ?, NULL, ?, ?)""",
                (jid, project_id, kind, state, now, now),
            )

    def set_job_state(self, jid: str, state: str, error: str | None = None) -> None:
        with self._lock, self._txn() as con:
            con.execute("UPDATE jobs SET state=?, error=?, updated_at=? WHERE id=?",
                        (state, error, _now(), jid))

    def get_job(self, jid: str) -> dict | None:
        with self._txn() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(row) if row is not None else None

    def list_jobs(self, project_id: str | None = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM jobs"
        params: tuple = ()
        if project_id is not None:
            query += " WHERE project_id=?"
            params = (project_id,)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._txn() as con:
            return [dict(r) for r in con.execute(query, (*params, limit)).fetchall()]
