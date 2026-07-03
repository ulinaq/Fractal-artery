"""Tests for the FastAPI back-end (needs the api extra + httpx)."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from perfusion.api.app import create_app
from perfusion.api.store import ProjectStore

SMALL_GRID = {"nx": 60, "ny": 70, "lx": 140, "ly": 160,
              "domain_limit": [-70, 70, 0, 160]}
FAST_PARAMS = {"r_min_frac": 0.5, "seed": 0}   # shallow tree -> quick solve


@pytest.fixture
def client(tmp_path):
    # each test gets an isolated on-disk store under its tmp_path
    return TestClient(create_app(store=ProjectStore(tmp_path)))


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


def test_tracer_animation(client):
    pid = client.post("/projects", json={"name": "flow", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 16}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    assert client.get(f"/projects/{pid}").json()["has_tracer"] is True

    info = client.get(f"/projects/{pid}/tracer").json()
    assert info["n_frames"] == 16
    assert len(info["frames"]) == 16
    assert len(info["times"]) == 16
    assert tuple(info["shape"]) == (60, 70)
    # the bolus builds up: the peak per-frame concentration is not in the first frame
    def frame_peak(f):
        return max((v for row in f for v in row if v is not None), default=0.0)

    peaks = [frame_peak(f) for f in info["frames"]]
    assert max(peaks) > 0
    assert peaks.index(max(peaks)) > 0


def test_tracer_gif_export(client):
    pytest.importorskip("imageio")
    pid = client.post("/projects", json={"name": "gif", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 8}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    gif = client.get(f"/projects/{pid}/tracer.gif")
    assert gif.status_code == 200
    assert gif.headers["content-type"] == "image/gif"
    assert gif.content[:6] in (b"GIF87a", b"GIF89a")


def test_image_upload_segments_a_network(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    from perfusion.io import DATA_DIR

    img = (DATA_DIR / "frog_art_low.png").read_bytes()
    pid = client.post("/projects", json={"name": "img", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    r = client.post(f"/projects/{pid}/image", content=img, params={"min_size": 30})
    assert r.status_code == 200

    segs = client.get(f"/projects/{pid}/network").json()["segments"]
    assert len(segs) > 10
    xs = [c for s in segs for c in (s["x0"], s["x1"])]
    ys = [c for s in segs for c in (s["y0"], s["y1"])]
    assert min(xs) >= -70                          # placed inside the domain window
    assert max(xs) <= 70
    assert min(ys) >= 0
    assert max(ys) <= 160
    # the segmented tree is usable by the solver
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_image_upload_rejects_bad_payload(client):
    pytest.importorskip("imageio")
    pid = client.post("/projects", json={"name": "bad", "grid": SMALL_GRID}).json()["id"]
    assert client.post(f"/projects/{pid}/image", content=b"").status_code == 400
    assert client.post(f"/projects/{pid}/image", content=b"not-an-image").status_code == 422


def test_frontend_is_served(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "perfusion" in root.text.lower()
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_mask_makes_the_solved_domain_organ_shaped(client):
    pid = client.post("/projects", json={"name": "mask", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    # a polygon inside the (-70,70,0,160) domain that clips its borders
    region = [[-60, 8], [60, 8], [60, 150], [-60, 150]]
    assert client.put(f"/projects/{pid}/mask", json={"regions": [region]}).json()["has_mask"]
    assert client.get(f"/projects/{pid}/mask").json()["regions"]

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    flat = [v for row in client.get(f"/projects/{pid}/results/field/pressure").json()["data"]
            for v in row]
    assert any(v is None for v in flat)       # border cells are inactive (masked out)
    assert any(v is not None for v in flat)   # the organ interior is solved


def test_projects_persist_across_store_reopen(tmp_path):
    # Run a project (with a hand-drawn tissue mask) through one store instance...
    c1 = TestClient(create_app(store=ProjectStore(tmp_path)))
    pid = c1.post("/projects", json={"name": "persist", "grid": SMALL_GRID,
                                     "params": FAST_PARAMS}).json()["id"]
    c1.put(f"/projects/{pid}/mask", json={"regions": [[[-60, 8], [60, 8], [60, 150], [-60, 150]]]})
    c1.post(f"/projects/{pid}/grow", json={"n_points": 80, "step": 6.0})
    jid = c1.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(c1, jid)["state"] == "done"

    # ...then reopen a *fresh* store at the same location (simulates a restart).
    c2 = TestClient(create_app(store=ProjectStore(tmp_path)))
    info = c2.get(f"/projects/{pid}").json()
    assert info["name"] == "persist"
    assert info["has_result"] is True
    assert info["has_mask"] is True               # tissue outline survived
    assert info["has_vessels"] is False           # grown, not hand-drawn
    assert c2.get(f"/projects/{pid}/mask").json()["regions"]
    assert c2.get(f"/projects/{pid}/results").json()["arterial_inflow"] > 0
    assert tuple(c2.get(f"/projects/{pid}/results/field/pressure").json()["shape"]) == (60, 70)
    assert len(c2.get(f"/projects/{pid}/network").json()["segments"]) > 0   # tree survived
    assert any(p["id"] == pid for p in c2.get("/projects").json())


def test_missing_project_and_result(client):
    assert client.get("/projects/nope").status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    pid = client.post("/projects", json={"name": "empty", "grid": SMALL_GRID}).json()["id"]
    assert client.get(f"/projects/{pid}/results").status_code == 409  # not run yet
