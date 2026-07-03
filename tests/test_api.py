"""Tests for the FastAPI back-end (needs the api extra + httpx)."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from perfusion.api.app import create_app

SMALL_GRID = {"nx": 60, "ny": 70, "lx": 140, "ly": 160,
              "domain_limit": [-70, 70, 0, 160]}
FAST_PARAMS = {"r_min_frac": 0.5, "seed": 0}   # shallow tree -> quick solve


@pytest.fixture
def client():
    return TestClient(create_app())


def _wait_for_job(client, jid, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/jobs/{jid}").json()
        if state["state"] in ("done", "error"):
            return state
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_list_projects(client):
    r = client.post("/projects", json={"name": "demo", "grid": SMALL_GRID,
                                       "params": FAST_PARAMS})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert client.get("/projects/" + pid).json()["name"] == "demo"
    assert any(p["id"] == pid for p in client.get("/projects").json())


def test_run_results_and_export(client):
    pid = client.post("/projects", json={"name": "run", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={"nonlinear": False}).json()["id"]
    state = _wait_for_job(client, jid)
    assert state["state"] == "done", state

    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["arterial_inflow"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)

    field = client.get(f"/projects/{pid}/results/field/pressure").json()
    assert tuple(field["shape"]) == (60, 70)

    export = client.get(f"/projects/{pid}/export")
    assert export.status_code == 200
    assert len(export.content) > 1000


def test_run_with_hand_drawn_vessels(client):
    pid = client.post("/projects", json={
        "name": "drawn",
        "grid": {"nx": 60, "ny": 60, "lx": 60, "ly": 60,
                 "domain_limit": [0, 60, 0, 60]},
    }).json()["id"]
    vessels = {
        "arteries": [{"points": [[10, 5], [10, 30]], "radius": 2.0},
                     {"points": [[10, 30], [5, 50]], "radius": 1.0},
                     {"points": [[10, 30], [25, 50]], "radius": 1.0}],
        "veins": [{"points": [[45, 55], [45, 30]], "radius": 2.0},
                  {"points": [[45, 30], [30, 8]], "radius": 1.0},
                  {"points": [[45, 30], [55, 8]], "radius": 1.0}],
    }
    assert client.put(f"/projects/{pid}/vessels", json=vessels).json()["has_vessels"]
    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    state = _wait_for_job(client, jid)
    assert state["state"] == "done", state
    assert client.get(f"/projects/{pid}/results").json()["arterial_inflow"] > 0


def test_grow_then_run(client):
    pid = client.post("/projects", json={"name": "grow", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    grown = client.post(f"/projects/{pid}/grow",
                        json={"n_points": 80, "seed": 0, "step": 6.0}).json()
    assert grown["n_segments"] > 0
    assert grown["n_terminals"] > 0

    # the grown tree is exposed as segment geometry for the front-end overlay
    net = client.get(f"/projects/{pid}/network").json()
    assert len(net["segments"]) == grown["n_segments"]
    assert {"x0", "y0", "x1", "y1", "r", "kind"} <= set(net["segments"][0])

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    state = _wait_for_job(client, jid)
    assert state["state"] == "done", state
    assert client.get(f"/projects/{pid}/results").json()["arterial_inflow"] > 0


def test_frontend_is_served(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "perfusion" in root.text.lower()
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_missing_project_and_result(client):
    assert client.get("/projects/nope").status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    pid = client.post("/projects", json={"name": "empty", "grid": SMALL_GRID}).json()["id"]
    assert client.get(f"/projects/{pid}/results").status_code == 409  # not run yet
