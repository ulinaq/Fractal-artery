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


def test_run_max_iter_surfaces_nonconvergence(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    # nonlinear solve capped at a single iteration -> should not converge and warn
    jid = client.post(f"/projects/{pid}/run",
                      json={"use_vessels": True, "nonlinear": True, "max_iter": 1}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert "notes" in summary
    assert any("converge" in n.lower() for n in summary["notes"])

    # a generous cap converges cleanly (no notes)
    jid = client.post(f"/projects/{pid}/run",
                      json={"use_vessels": True, "nonlinear": True, "max_iter": 200}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    assert "notes" not in client.get(f"/projects/{pid}/results").json()


def test_direct_connection_model_runs_and_renders(client):
    pid = client.post("/projects", json={"name": "direct", "params": {"p_in": 40, "p_out": 0},
                                         "grid": SMALL_GRID}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={"model": "direct", "rho_c": 0}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["model"] == "direct"
    assert summary["n_connections"] > 0
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"])   # mass conserved
    d = client.get(f"/projects/{pid}/direct").json()
    assert len(d["artery"]) > 0
    assert len(d["vein"]) > 0
    assert len(d["connections"]) == summary["n_connections"]
    assert all(len(seg) == 5 for seg in d["artery"][:3])       # [x0,y0,x1,y1,flow]
    # a hybrid project has no direct blob
    assert client.get(f"/projects/{pid}/results/field/pressure").status_code in (404, 409)


def test_direct_network_tracer(client):
    pid = client.post("/projects", json={"name": "nettracer", "params": {"p_in": 40, "p_out": 0},
                                         "grid": SMALL_GRID}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={"model": "direct", "rho_c": 0}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    # no network tracer until we ask for one
    assert client.get(f"/projects/{pid}/network-tracer").status_code == 409
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 20}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    t = client.get(f"/projects/{pid}/network-tracer").json()
    assert t["n_frames"] == 20
    n_elem = len(t["lines"])
    assert n_elem > 0
    assert len(t["frames"]) == 20
    assert all(len(f) == n_elem for f in t["frames"])
    assert set(t["kind"]) <= {0, 1, 2}
    assert all(len(ln) == 4 for ln in t["lines"][:3])          # [x0,y0,x1,y1]
    # the bolus lights up at least some elements
    assert max(max(f) for f in t["frames"]) > 0.5


def test_run_zero_pressure_drop_gives_clear_error(client):
    pid = client.post("/projects", json={"name": "flat", "grid": SMALL_GRID,
                                         "params": {"p_in": 5, "p_out": 5}}).json()["id"]
    client.put(f"/projects/{pid}/vessels", json={
        "arteries": [{"points": [[0, 10], [0, 60]], "radius": 2.0}], "veins": []})
    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    final = _wait_for_job(client, jid)
    assert final["state"] == "error"
    assert "p_in equals p_out" in final["error"]


def test_run_disconnected_mask_gives_clear_error(client):
    pid = client.post("/projects", json={"name": "split", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    client.put(f"/projects/{pid}/vessels", json={
        "arteries": [{"points": [[-40, 20], [-40, 60]], "radius": 2.0}], "veins": []})
    # two separate tissue blobs -> disconnected Darcy domain
    client.put(f"/projects/{pid}/mask", json={"regions": [
        [[-60, 10], [-20, 10], [-20, 70], [-60, 70]],
        [[20, 10], [60, 10], [60, 70], [20, 70]]]})
    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    final = _wait_for_job(client, jid)
    assert final["state"] == "error"
    assert "disconnected regions" in final["error"]


def test_frontend_version_endpoint(client):
    h = client.get("/health").json()
    assert "build" in h                            # health carries the build too
    v = client.get("/frontend-version").json()
    assert isinstance(v["build"], str)
    assert v["build"]
    assert v["build"] == h["build"]                # same hash from both


def test_explore_evaluate_returns_field_tree_objective(client):
    r = client.post("/explore", json={"a_gamma": 1.4, "v_gamma": 1.2, "kind": "flux_std",
                                      "field": "flux", "nx": 40, "ny": 46})
    assert r.status_code == 200
    d = r.json()
    assert d["objective"] > 0
    assert d["field"]["name"] == "flux"
    assert d["field"]["shape"] == [40, 46]
    assert len(d["segments"]) > 0
    assert len(d["segments"][0]) == 6            # x0,y0,x1,y1,r,kind
    assert {s[5] for s in d["segments"]} == {0, 1}   # artery + vein


def test_explore_optimize_improves(client):
    o = client.post("/explore/optimize", json={"kind": "flux_std", "method": "local",
                                               "maxiter": 6, "nx": 40, "ny": 46}).json()
    assert o["best"] <= o["start"] + 1e-9
    assert o["n_eval"] > 0
    assert 1.0 <= o["a_gamma"] <= 1.9


def test_explorer_page_served(client):
    assert client.get("/explorer.html").status_code == 200
    assert client.get("/explorer.js").status_code == 200


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


def test_lesions_set_get_run_and_roundtrip(client):
    pid = client.post("/projects", json={"name": "lesions", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/lesions").json()["lesions"] == []
    lesion = {"polygon": [[-20, 60], [20, 60], [20, 100], [-20, 100]], "permeability": 0.05}
    client.put(f"/projects/{pid}/lesions", json={"lesions": [lesion]})
    got = client.get(f"/projects/{pid}/lesions").json()["lesions"]
    assert got[0]["permeability"] == 0.05
    assert got[0]["polygon"][0] == [-20, 60]
    # the heterogeneous K field feeds a normal solve
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    # document round-trip carries the lesions into a fresh project
    doc = client.get(f"/projects/{pid}/document").json()
    assert doc["lesions"][0]["permeability"] == 0.05
    pid2 = client.post("/projects/import", json=doc).json()["id"]
    assert client.get(f"/projects/{pid2}/lesions").json()["lesions"][0]["permeability"] == 0.05


def test_setting_lesions_invalidates_result(client):
    pid = client.post("/projects", json={"name": "inv", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    assert client.get(f"/projects/{pid}").json()["has_result"] is True
    client.put(f"/projects/{pid}/lesions",
               json={"lesions": [{"polygon": [[-10, 60], [10, 60], [0, 90]], "permeability": 0.1}]})
    assert client.get(f"/projects/{pid}").json()["has_result"] is False   # stale result cleared


def test_document_roundtrip_basic(client):
    # a plain project with params, a tissue region and I/O points
    pid = client.post("/projects", json={"name": "my work", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    region = [[-60, 8], [60, 8], [60, 150], [-60, 150]]
    client.put(f"/projects/{pid}/mask", json={"regions": [region]})
    client.put(f"/projects/{pid}/io", json={"inlets": [[0, 20]], "outlets": [[0, 140]]})

    doc = client.get(f"/projects/{pid}/document")
    assert doc.status_code == 200
    assert "attachment" in doc.headers["content-disposition"]
    assert doc.headers["content-disposition"].endswith('my_work.perfusion.json"')
    body = doc.json()
    assert body["version"] == 1
    assert body["name"] == "my work"

    info = client.post("/projects/import", json=body).json()
    assert info["id"] != pid                        # a fresh project
    pid2 = info["id"]
    assert client.get(f"/projects/{pid2}/mask").json()["regions"] == [region]
    io2 = client.get(f"/projects/{pid2}/io").json()
    assert io2["inlets"] == [[0, 20]]
    assert io2["outlets"] == [[0, 140]]
    got = client.get(f"/projects/{pid2}").json()
    assert got["params"]["r_min_frac"] == 0.5
    assert got["grid"]["nx"] == 60


def test_document_roundtrip_frog_networks_and_mask(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    body = client.get(f"/projects/{pid}/document").json()
    # the segmented trees and raster mask travel inside the JSON
    assert len(body["artery_network"]["node_x"]) > 10
    assert len(body["vein_network"]["seg_start"]) > 10
    assert body["mask"]["shape"] == [120, 140]

    pid2 = client.post("/projects/import", json=body).json()["id"]
    # the restored trees are still there and the project solves + conserves mass
    kinds = {s["kind"] for s in client.get(f"/projects/{pid2}/network").json()["segments"]}
    assert kinds == {"artery", "vein"}
    jid = client.post(f"/projects/{pid2}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid2}/results").json()
    assert summary["arterial_inflow"] == pytest.approx(summary["venous_outflow"], rel=1e-6)


def test_import_rejects_garbage(client):
    assert client.post("/projects/import", json={"grid": "not a grid"}).status_code == 422


def test_editable_move_node_and_resize(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    ed = client.get(f"/projects/{pid}/editable").json()
    assert {k for k in ed if ed[k]} == {"artery", "vein"}
    a = ed["artery"]
    n_nodes, n_segs = len(a["node_x"]), len(a["seg_radius"])

    # move node 0 by +12 in x and double every radius
    nx_ = list(a["node_x"])
    nx_[0] += 12.0
    radii = [2 * r for r in a["seg_radius"]]
    put = client.put(f"/projects/{pid}/editable",
                     json={"artery": {"node_x": nx_, "node_y": a["node_y"],
                                      "seg_radius": radii}})
    assert put.status_code == 200

    ed2 = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert ed2["node_x"][0] == pytest.approx(a["node_x"][0] + 12.0)
    assert ed2["seg_radius"] == pytest.approx(radii)
    assert len(ed2["node_x"]) == n_nodes          # topology preserved
    assert len(ed2["seg_radius"]) == n_segs

    # the edited network still solves and conserves mass
    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    s = client.get(f"/projects/{pid}/results").json()
    assert s["arterial_inflow"] == pytest.approx(s["venous_outflow"], rel=1e-6)


def test_editable_rejects_wrong_counts(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    bad = client.put(f"/projects/{pid}/editable",
                     json={"artery": {"node_x": [1.0, 2.0], "node_y": [1.0, 2.0],
                                      "seg_radius": a["seg_radius"]}})
    assert bad.status_code == 422


def test_editable_add_and_delete_topology(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    n0, m0 = len(a["node_x"]), len(a["seg_radius"])

    # ADD: a new node joined to node 3 by a new segment
    add = {"node_x": a["node_x"] + [a["node_x"][3] + 5.0],
           "node_y": a["node_y"] + [a["node_y"][3] + 5.0],
           "seg_start": a["seg_start"] + [3], "seg_end": a["seg_end"] + [n0],
           "seg_radius": a["seg_radius"] + [1.5],
           "inlet_nodes": a["inlet_nodes"], "root_node": a["root_node"]}
    assert client.put(f"/projects/{pid}/editable", json={"artery": add}).status_code == 200
    a2 = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert len(a2["node_x"]) == n0 + 1
    assert len(a2["seg_radius"]) == m0 + 1

    # DELETE: drop a leaf's feeding segment -> the leaf becomes isolated and is pruned
    deg = [0] * len(a2["node_x"])
    for s in a2["seg_start"] + a2["seg_end"]:
        deg[s] += 1
    leaf = next(i for i in range(len(deg)) if deg[i] == 1 and i not in a2["inlet_nodes"])
    keep = [k for k in range(len(a2["seg_start"]))
            if a2["seg_start"][k] != leaf and a2["seg_end"][k] != leaf]
    dele = {"node_x": a2["node_x"], "node_y": a2["node_y"],
            "seg_start": [a2["seg_start"][k] for k in keep],
            "seg_end": [a2["seg_end"][k] for k in keep],
            "seg_radius": [a2["seg_radius"][k] for k in keep],
            "inlet_nodes": a2["inlet_nodes"], "root_node": a2["root_node"]}
    assert client.put(f"/projects/{pid}/editable", json={"artery": dele}).status_code == 200
    a3 = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert len(a3["node_x"]) == n0        # isolated leaf dropped, back to the start count
    assert len(a3["seg_radius"]) == m0

    # the rebuilt tree still solves and conserves mass
    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    s = client.get(f"/projects/{pid}/results").json()
    assert s["arterial_inflow"] == pytest.approx(s["venous_outflow"], rel=1e-6)


def test_vectorize_handdrawn_to_network_keeps_per_edge_radius(client):
    pid = client.post("/projects", json={"name": "draw", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    # a Y-shaped artery: three strokes with distinct radii sharing a junction
    client.put(f"/projects/{pid}/vessels", json={"arteries": [
        {"points": [[0, 10], [0, 40]], "radius": 3.0},
        {"points": [[0, 40], [-20, 80]], "radius": 1.5},
        {"points": [[0, 40], [20, 80]], "radius": 2.0}], "veins": []})
    assert client.get(f"/projects/{pid}/editable").json() == {"artery": None, "vein": None}

    assert client.post(f"/projects/{pid}/vectorize").status_code == 200
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert len(a["node_x"]) == 4                       # trunk base, junction, two tips
    assert len(a["seg_radius"]) == 3
    assert sorted(round(r, 1) for r in a["seg_radius"]) == [1.5, 2.0, 3.0]   # per-edge radii kept
    # the shared branch point is one node, not three coincident ones (a real tree)
    junction = [i for i in range(4) if abs(a["node_x"][i]) < 0.6 and abs(a["node_y"][i] - 40) < 0.6]
    assert len(junction) == 1
    # polylines are consumed; it now solves as a network
    assert client.get(f"/projects/{pid}").json()["has_vessels"] is False
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_grow_vein_target_and_shape_preserving_base(client):
    pid = client.post("/projects", json={"name": "grow", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    # a bent artery (3 points) and a bent vein
    client.put(f"/projects/{pid}/vessels", json={
        "arteries": [{"points": [[0, 10], [-15, 40], [0, 70]], "radius": 3.0}],
        "veins": [{"points": [[10, 10], [25, 40], [10, 70]], "radius": 2.5}]})

    ga = client.post(f"/projects/{pid}/grow",
                     json={"target": "artery", "n_points": 60, "seed": 0}).json()
    gv = client.post(f"/projects/{pid}/grow",
                     json={"target": "vein", "n_points": 60, "seed": 1}).json()
    assert ga["n_segments"] > 2
    assert gv["n_segments"] > 2

    # the base kept the drawn bend: a node near the artery's mid-point (-15, 40)
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert any(abs(x + 15) < 1.0 and abs(y - 40) < 1.0
               for x, y in zip(a["node_x"], a["node_y"], strict=True))
    # both trees exist and the coupled system solves + conserves mass
    assert client.get(f"/projects/{pid}/editable").json()["vein"] is not None
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    s = client.get(f"/projects/{pid}/results").json()
    assert s["arterial_inflow"] == pytest.approx(s["venous_outflow"], rel=1e-6)


def test_transit_metrics_in_summary_and_endpoint(client):
    pid = client.post("/projects", json={"name": "ttd", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    for key in ("mtt", "cth", "rd"):
        assert key in summary
    assert summary["mtt"] > 0
    assert summary["rd"] == pytest.approx(summary["cth"] / summary["mtt"], rel=1e-6)

    t = client.get(f"/projects/{pid}/transit", params={"bins": 30}).json()
    assert t["mtt"] == pytest.approx(summary["mtt"])
    assert 0.0 <= t["oef"] <= 1.0
    c = t["histogram"]["centres"]
    d = t["histogram"]["density"]
    assert len(c) == len(d) == 30
    width = c[1] - c[0]
    assert sum(d) * width == pytest.approx(1.0, rel=1e-6)     # normalised density


def test_tracer_kinetics_fit(client):
    pid = client.post("/projects", json={"name": "tkm", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/kinetics").status_code == 409   # no tracer yet
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 24}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    k = client.get(f"/projects/{pid}/kinetics", params={"model": "1CM"}).json()
    assert k["model"] == "1CM"
    assert k["Fp"] is not None
    assert k["Fp"] > 0
    assert len(k["times"]) == len(k["measured"]) == len(k["fitted"]) == len(k["aif"])


def test_transient_pulsatile_damping(client):
    pid = client.post("/projects", json={"name": "tr", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/transient").status_code == 409   # no result yet
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    d = client.get(f"/projects/{pid}/transient",
                   params={"period": 1.0, "pulse": 0.5, "compliance": 0.05}).json()
    assert d["R"] > 0
    assert d["C"] > 0
    assert len(d["t"]) == len(d["p_in"]) == len(d["q_tissue"])
    assert d["tissue_pulsatility"] < d["inlet_pulsatility"]      # compliance damps the pulse


def test_kinetic_maps_endpoint(client):
    pid = client.post("/projects", json={"name": "maps", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/kinetics/maps").status_code == 409   # no tracer
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 24}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    fp = client.get(f"/projects/{pid}/kinetics/maps", params={"field": "Fp"}).json()
    assert fp["name"] == "Fp"
    assert tuple(fp["shape"]) == (SMALL_GRID["nx"], SMALL_GRID["ny"])
    assert any(v is not None for row in fp["data"] for v in row)   # some perfused voxels
    assert client.get(f"/projects/{pid}/kinetics/maps", params={"field": "nope"}).status_code == 404
    # deconvolution method exposes CBV / MTT
    mtt = client.get(f"/projects/{pid}/kinetics/maps",
                     params={"field": "MTT", "method": "deconv"})
    assert mtt.status_code == 200
    assert tuple(mtt.json()["shape"]) == (SMALL_GRID["nx"], SMALL_GRID["ny"])
    # MTT is a deconv-only field; unknown method rejected
    assert client.get(f"/projects/{pid}/kinetics/maps",
                      params={"field": "MTT", "method": "semi"}).status_code == 404
    assert client.get(f"/projects/{pid}/kinetics/maps",
                      params={"field": "Fp", "method": "bogus"}).status_code == 404


def test_uq_endpoint(client):
    pid = client.post("/projects", json={"name": "uq", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    d = client.post(f"/projects/{pid}/uq",
                    json={"n_samples": 12, "params": {"p_in": [5, 20], "mu": [0.8, 1.5]}}).json()
    assert d["n_samples"] == 12
    assert "arterial_inflow" in d["qoi_names"]
    assert d["std"]["arterial_inflow"] > 0
    assert d["sensitivity"]["arterial_inflow"]["p_in"] > 0.5     # inflow tracks inlet pressure
    # an unknown uncertain parameter is a clear 422
    assert client.post(f"/projects/{pid}/uq",
                       json={"n_samples": 4, "params": {"bogus": [0, 1]}}).status_code == 422
    # variance-based Sobol mode returns first-order + total-effect indices
    sob = client.post(f"/projects/{pid}/uq",
                      json={"n_samples": 24, "method": "sobol",
                            "params": {"p_in": [5, 20], "mu": [0.8, 1.5]}}).json()
    assert sob["method"] == "sobol"
    s1 = sob["first_order"]["arterial_inflow"]
    st = sob["total_effect"]["arterial_inflow"]
    assert st["p_in"] >= s1["p_in"] - 1e-9
    assert s1["p_in"] == max(s1.values())


def test_adapt_endpoint_moves_toward_murray(client):
    pid = client.post("/projects", json={"name": "adapt", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    info = client.post(f"/projects/{pid}/adapt",
                       json={"target_exponent": 3.0, "rate": 0.05, "n_iter": 60}).json()
    assert info["moved_toward_target"] is True
    assert abs(info["murray_final"] - 3.0) < abs(info["murray_start"] - 3.0)
    assert info["n_iter"] >= 1
    # the adapted tree is stored as the project's network and is solvable
    net = client.get(f"/projects/{pid}/network").json()
    assert any(s["kind"] == "artery" for s in net["segments"])
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_morphometry_endpoint(client):
    pid = client.post("/projects", json={"name": "morph", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    d = client.get(f"/projects/{pid}/morphometry", params={"target": "artery"}).json()
    assert d["target"] == "artery"
    assert d["max_order"] >= 2
    assert 1.0 < d["bifurcation_ratio"] < 6.0
    assert d["counts"]                                   # per-order segment counts present


def test_growth_hotspots_persist_and_feed_growth(client):
    pid = client.post("/projects", json={"name": "grow-hot", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/hotspots").json()["hotspots"] == []
    hot = {"polygon": [[5, 20], [65, 20], [65, 150], [5, 150]], "weight": 12.0}
    client.put(f"/projects/{pid}/hotspots", json={"hotspots": [hot]})
    assert client.get(f"/projects/{pid}/hotspots").json()["hotspots"][0]["weight"] == 12.0
    # the demand map feeds a valid data-constrained grow (bias magnitude is covered by
    # the growth/MCP unit tests; here we assert the endpoint threads it through cleanly)
    info = client.post(f"/projects/{pid}/grow",
                       json={"target": "artery", "n_points": 400, "seed": 1}).json()
    assert info["n_segments"] > 0
    segs = client.get(f"/projects/{pid}/network").json()["segments"]
    assert any(s["kind"] == "artery" for s in segs)
    # hotspots round-trip through the document
    doc = client.get(f"/projects/{pid}/document").json()
    assert doc["hotspots"][0]["weight"] == 12.0
    pid2 = client.post("/projects/import", json=doc).json()["id"]
    assert client.get(f"/projects/{pid2}/hotspots").json()["hotspots"][0]["weight"] == 12.0


def test_anisotropic_permeability_roundtrips_and_solves(client):
    grid = {**SMALL_GRID, "permeability": 1.0, "permeability_y": 5.0}
    pid = client.post("/projects", json={"name": "aniso", "grid": grid,
                                         "params": FAST_PARAMS}).json()["id"]
    got = client.get(f"/projects/{pid}").json()
    assert got["grid"]["permeability_y"] == 5.0
    # the anisotropic bed still solves
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    # and survives a document round-trip into a fresh project
    doc = client.get(f"/projects/{pid}/document").json()
    assert doc["grid"]["permeability_y"] == 5.0
    pid2 = client.post("/projects/import", json=doc).json()["id"]
    assert client.get(f"/projects/{pid2}").json()["grid"]["permeability_y"] == 5.0


def test_multicompartment_endpoint(client):
    pid = client.post("/projects", json={"name": "mc", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    perf = client.get(f"/projects/{pid}/multicompartment",
                      params={"field": "perfusion", "n": 3, "beta": 1.0}).json()
    assert perf["name"] == "perfusion"
    assert perf["n"] == 3
    assert tuple(perf["shape"]) == (SMALL_GRID["nx"], SMALL_GRID["ny"])
    vals = [v for row in perf["data"] for v in row if v is not None]
    assert vals
    assert sum(vals) > 0                   # net arteriole→capillary perfusion
    # a compartment pressure field
    p0 = client.get(f"/projects/{pid}/multicompartment", params={"field": "comp0", "n": 3})
    assert p0.status_code == 200
    # an out-of-range compartment / bad field is a clear 400
    assert client.get(f"/projects/{pid}/multicompartment",
                      params={"field": "comp9", "n": 3}).status_code == 400


def test_oxygen_endpoint(client):
    pid = client.post("/projects", json={"name": "o2", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/oxygen").status_code == 409   # no result yet
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    sat = client.get(f"/projects/{pid}/oxygen", params={"field": "saturation", "k": 1.0}).json()
    assert sat["name"] == "saturation"
    assert tuple(sat["shape"]) == (SMALL_GRID["nx"], SMALL_GRID["ny"])
    assert 0.0 < sat["oef"] < 1.0
    vals = [v for row in sat["data"] for v in row if v is not None]
    assert vals
    assert max(vals) <= 1.0 + 1e-9
    assert min(vals) >= 0.0
    # more extraction with larger k
    hi = client.get(f"/projects/{pid}/oxygen", params={"field": "saturation", "k": 5.0}).json()
    assert hi["oef"] > sat["oef"]
    assert client.get(f"/projects/{pid}/oxygen", params={"field": "nope"}).status_code == 404


def test_transit_409_without_field_result(client):
    pid = client.post("/projects", json={"name": "d", "params": {"p_in": 40, "p_out": 0},
                                         "grid": SMALL_GRID}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={"model": "direct"}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    assert client.get(f"/projects/{pid}/transit").status_code == 409   # no grid field


def test_project_view3d_before_and_after_run(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    pid = client.post("/projects/frog").json()["id"]
    before = client.get(f"/projects/{pid}/view3d").json()
    assert before["has_result"] is False
    assert len(before["segments"]) > 0            # trees always show (the geometry)
    assert before["pressure"]["voxels"] == []     # no field until solved

    jid = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    after = client.get(f"/projects/{pid}/view3d").json()
    assert after["has_result"] is True
    assert len(after["pressure"]["voxels"]) > 0   # pressure relief voxels
    assert after["domain"][5] > 0                 # non-zero z-extent (relief)
    assert {s[7] for s in after["segments"]} == {0, 1}   # artery + vein


def test_grow_rejects_bad_target(client):
    pid = client.post("/projects", json={"name": "g", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.post(f"/projects/{pid}/grow", json={"target": "nope"}).status_code == 422


def test_vectorize_without_vessels_400(client):
    pid = client.post("/projects", json={"name": "empty", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.post(f"/projects/{pid}/vectorize").status_code == 400


def test_editable_absent_without_networks(client):
    pid = client.post("/projects", json={"name": "empty", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    ed = client.get(f"/projects/{pid}/editable").json()
    assert ed == {"artery": None, "vein": None}
    # nothing to edit -> 404
    assert client.put(f"/projects/{pid}/editable",
                      json={"artery": {"node_x": [], "node_y": [], "seg_radius": []}}
                      ).status_code == 404


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
