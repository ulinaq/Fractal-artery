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

    csv = client.get(f"/projects/{pid}/export", params={"fmt": "csv"})
    assert csv.status_code == 200
    assert csv.text.splitlines()[0] == "x,y,pressure,flux,tof"

    vtk = client.get(f"/projects/{pid}/export", params={"fmt": "vtk"})
    assert vtk.status_code == 200
    assert vtk.text.lstrip().startswith("<?xml")
    assert 'type="ImageData"' in vtk.text

    assert client.get(f"/projects/{pid}/export", params={"fmt": "bogus"}).status_code == 422

    # one-zip bundle of everything
    import io as _io
    import zipfile

    bundle = client.get(f"/projects/{pid}/bundle")
    assert bundle.status_code == 200
    with zipfile.ZipFile(_io.BytesIO(bundle.content)) as z:
        assert {"results.npz", "fields.csv", "fields.vti", "summary.json",
                "meta.json"} <= set(z.namelist())


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

    # the CCO method is also available and produces a usable tree
    cco = client.post(f"/projects/{pid}/grow",
                      json={"method": "cco", "n_points": 60, "seed": 1}).json()
    assert cco["n_segments"] > 0
    assert client.post(f"/projects/{pid}/grow",
                       json={"method": "bogus"}).status_code == 422

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


def test_image_upload_artery_and_vein(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    from perfusion.io import DATA_DIR

    art = (DATA_DIR / "frog_art_low.png").read_bytes()
    ven = (DATA_DIR / "frog_vein.png").read_bytes()
    pid = client.post("/projects", json={"name": "av", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.post(f"/projects/{pid}/image", content=art,
                       params={"kind": "artery", "min_size": 30}).status_code == 200
    assert client.post(f"/projects/{pid}/image", content=ven,
                       params={"kind": "vein", "min_size": 30}).status_code == 200
    # both segmented trees show up in the overlay geometry
    kinds = {s["kind"] for s in client.get(f"/projects/{pid}/network").json()["segments"]}
    assert kinds == {"artery", "vein"}
    # and the coupled system solves + conserves mass
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)


def test_image_upload_rejects_bad_payload(client):
    pytest.importorskip("imageio")
    pid = client.post("/projects", json={"name": "bad", "grid": SMALL_GRID}).json()["id"]
    assert client.post(f"/projects/{pid}/image", content=b"").status_code == 400
    assert client.post(f"/projects/{pid}/image", content=b"not-an-image").status_code == 422
    assert client.post(f"/projects/{pid}/image", content=b"x",
                       params={"kind": "bogus"}).status_code == 422


def test_jobs_are_listed_and_persist(tmp_path):
    c1 = TestClient(create_app(store=ProjectStore(tmp_path)))
    pid = c1.post("/projects", json={"name": "jobs", "grid": SMALL_GRID,
                                     "params": FAST_PARAMS}).json()["id"]
    jid = c1.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(c1, jid)["state"] == "done"

    listed = c1.get("/jobs").json()
    assert any(j["id"] == jid and j["kind"] == "run" for j in listed)
    assert c1.get(f"/jobs?project_id={pid}").json()[0]["id"] == jid

    # a finished job survives a "restart" (fresh store at the same location)
    c2 = TestClient(create_app(store=ProjectStore(tmp_path)))
    job = c2.get(f"/jobs/{jid}").json()
    assert job["state"] == "done"
    assert job["created_at"]
    assert job["updated_at"]


def test_stale_jobs_are_reaped_on_restart(tmp_path):
    store = ProjectStore(tmp_path)
    store.create_job("stuck", "p1", "run", state="running")   # never finished
    # reopening the store (a restart) marks in-flight jobs interrupted
    reopened = ProjectStore(tmp_path)
    assert reopened.get_job("stuck")["state"] == "interrupted"


def test_multiple_inlets_and_outlets(client):
    G = {"nx": 60, "ny": 70, "lx": 100, "ly": 120, "domain_limit": [0, 100, 0, 120]}
    pid = client.post("/projects", json={"name": "io", "grid": G,
                                         "params": FAST_PARAMS}).json()["id"]
    vessels = {
        "arteries": [{"points": p, "radius": 2.0} for p in
                     ([[50, 5], [50, 30]], [[50, 30], [20, 60]],
                      [[50, 30], [50, 65]], [[50, 30], [80, 60]])],
        "veins": [{"points": p, "radius": 2.0} for p in
                  ([[50, 115], [50, 90]], [[50, 90], [20, 60]],
                   [[50, 90], [50, 55]], [[50, 90], [80, 60]])],
    }
    client.put(f"/projects/{pid}/vessels", json=vessels)
    # two arterial inlets, two venous outlets
    assert client.put(f"/projects/{pid}/io",
                      json={"inlets": [[20, 60], [80, 60]],
                            "outlets": [[20, 60], [50, 55]]}).status_code == 200
    assert len(client.get(f"/projects/{pid}/io").json()["inlets"]) == 2

    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["arterial_inflow"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)


def test_separate_feeding_trees(client):
    # two disconnected arterial trees + two venous trees, one shared Darcy bed
    G = {"nx": 60, "ny": 70, "lx": 100, "ly": 120, "domain_limit": [0, 100, 0, 120]}
    pid = client.post("/projects", json={"name": "multi", "grid": G,
                                         "params": FAST_PARAMS}).json()["id"]
    arteries = [[[20, 6], [20, 30]], [[20, 30], [8, 60]], [[20, 30], [32, 60]],
                [[80, 6], [80, 30]], [[80, 30], [68, 60]], [[80, 30], [92, 60]]]
    veins = [[[20, 114], [20, 90]], [[20, 90], [8, 60]], [[20, 90], [32, 60]],
             [[80, 114], [80, 90]], [[80, 90], [68, 60]], [[80, 90], [92, 60]]]
    client.put(f"/projects/{pid}/vessels", json={
        "arteries": [{"points": p, "radius": 2.0} for p in arteries],
        "veins": [{"points": p, "radius": 2.0} for p in veins]})
    # mark an inlet on only ONE artery tree; the other must be auto-covered (not singular)
    client.put(f"/projects/{pid}/io", json={"inlets": [[8, 60]], "outlets": [[8, 60]]})

    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["arterial_inflow"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)


def test_io_points_persist_across_store_reopen(tmp_path):
    c1 = TestClient(create_app(store=ProjectStore(tmp_path)))
    pid = c1.post("/projects", json={"name": "io", "grid": SMALL_GRID}).json()["id"]
    c1.put(f"/projects/{pid}/io", json={"inlets": [[0, 80], [10, 90]], "outlets": [[0, 20]]})
    c2 = TestClient(create_app(store=ProjectStore(tmp_path)))
    io = c2.get(f"/projects/{pid}/io").json()
    assert io["inlets"] == [[0, 80], [10, 90]]
    assert io["outlets"] == [[0, 20]]


def test_frog_example_project(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    info = client.post("/projects/frog").json()
    assert info["has_mask"] is True                # tongue raster mask
    assert info["params"]["p_in"] == 40.0          # UI can reflect the project's settings
    assert info["grid"]["nx"] == 120
    pid = info["id"]
    kinds = {s["kind"] for s in client.get(f"/projects/{pid}/network").json()["segments"]}
    assert kinds == {"artery", "vein"}             # segmented arterial + venous trees

    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["arterial_inflow"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)


def test_frontend_is_served(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "perfusion" in root.text.lower()
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/viewer3d.html").status_code == 200      # 3-D viewer page
    assert client.get("/viewer3d.js").status_code == 200


def test_demo3d_payload(client):
    d = client.get("/demo3d", params={"gen": 4}).json()
    assert len(d["shape"]) == 3                                # (nx, ny, nz)
    assert len(d["domain"]) == 6
    assert len(d["segments"]) > 0
    seg = d["segments"][0]
    assert len(seg) == 8                                       # x0..z1, r, kind
    assert seg[7] in (0, 1)
    vox = d["pressure"]["voxels"]
    assert len(vox) > 0
    assert len(vox[0]) == 4                                    # x, y, z, pressure
    assert d["pressure"]["min"] <= d["pressure"]["max"]


def test_demo3d_frog_relief(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    d = client.get("/demo3d", params={"model": "frog"}).json()
    assert len(d["domain"]) == 6
    assert d["domain"][5] > 0                                  # pressure-relief z-extent
    assert {s[7] for s in d["segments"]} == {0, 1}             # artery + vein trees
    assert len(d["pressure"]["voxels"]) > 0


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
