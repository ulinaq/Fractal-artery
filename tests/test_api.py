"""Tests for the FastAPI back-end (needs the api extra + httpx)."""

import time

import numpy as np
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
    own = [sg for sg in net["segments"] if not sg.get("default")]
    assert len(own) == grown["n_segments"]
    assert net["defaults"] == ["vein"]          # the vein a run would use is the default
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


def test_tracer_mp4_export(client):
    pid = client.post("/projects", json={"name": "mp4", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    assert client.get(f"/projects/{pid}/tracer.mp4").status_code == 409   # no tracer yet
    pytest.importorskip("imageio_ffmpeg")
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 8}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    mp4 = client.get(f"/projects/{pid}/tracer.mp4")
    assert mp4.status_code == 200
    assert mp4.headers["content-type"] == "video/mp4"
    assert mp4.content[4:8] == b"ftyp"                 # ISO base-media container


def test_3d_project_tracer_video(client):
    """A 3-D project's flow video: GIF and MP4 render the bolus as a rotating
    maximum-intensity projection (the GIF has one frame per tracer frame)."""
    imageio = pytest.importorskip("imageio.v2")
    import io as stdio

    pid = _explorer_project(client, 3)["id"]
    assert client.get(f"/projects/{pid}/tracer.gif").status_code == 409   # no tracer yet
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 8}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    assert client.get(f"/projects/{pid}/view3d").json()["has_tracer"]

    gif = client.get(f"/projects/{pid}/tracer.gif?spin=false")
    assert gif.status_code == 200, gif.text
    frames = imageio.mimread(stdio.BytesIO(gif.content), format="GIF")
    assert len(frames) == 8
    assert frames[0].shape[:2] == (480, 480)
    pytest.importorskip("imageio_ffmpeg")
    mp4 = client.get(f"/projects/{pid}/tracer.mp4")
    assert mp4.status_code == 200
    assert mp4.content[4:8] == b"ftyp"


def test_image_upload_segments_a_network(client):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio")
    from perfusion.io import DATA_DIR

    img = (DATA_DIR / "frog_art_low.png").read_bytes()
    pid = client.post("/projects", json={"name": "img", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    r = client.post(f"/projects/{pid}/image", content=img, params={"min_size": 30})
    assert r.status_code == 200

    segs = [sg for sg in client.get(f"/projects/{pid}/network").json()["segments"]
            if not sg.get("default")]
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


@pytest.mark.parametrize("dim", [2, 3])
def test_explore_optimize_tunes_the_organ_setup_and_feeds_the_simulation(client, dim):
    """Optimise is organ-aware (the chosen shape, its roots, the fill) in 2-D and 3-D,
    and the optimised setup becomes a runnable project for the simulation page."""
    setup = {"dim": dim, "shape": "kidney", "fill": "cco", "n_terminals": 20,
             "generations": 3, "nx": 16 if dim == 3 else 50, "ny": 58, "kind": "tof_std"}
    o = client.post("/explore/optimize", json={**setup, "maxiter": 5}).json()
    assert o["dim"] == dim
    assert o["n_eval"] > 0
    assert o["best"] <= o["start"] + 1e-9                # never worse than the input
    assert 1.0 <= o["a_gamma"] <= 1.9
    assert ("twist" in o["free"]) == (dim == 3)
    assert len(o["history"]) == o["n_eval"]

    best = {**setup, "length_to_radius": o["length_to_radius"], "exponent": o["exponent"],
            "twist": o["twist"],
            "artery": {"gamma": o["a_gamma"]},
            "vein": {"gamma": o["v_gamma"], "radius": o["v_radius"], "angle": -90.0}}
    grown = client.post("/explore/grow", json=best).json()
    if dim == 2:                  # 'best' is scored at the user's own resolution
        assert grown["objective"] == pytest.approx(o["best"], rel=1e-6)
    made = client.post("/explore/project", json={**best, "name": "optimised"}).json()
    assert made["open"] == (f"/?project={made['id']}" if dim == 2
                            else f"/viewer3d.html?project={made['id']}")
    jid = client.post(f"/projects/{made['id']}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_explore_optimize_rejects_bad_input(client):
    assert client.post("/explore/optimize", json={"kind": "nope"}).status_code == 422
    assert client.post("/explore/optimize",
                       json={"free": ["not_a_param"]}).status_code == 422


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


def test_calibrate_endpoint_recovers_permeability(client):
    pid = client.post("/projects", json={"name": "cal",
                                         "grid": {**SMALL_GRID, "permeability": 2.0},
                                         "params": FAST_PARAMS}).json()["id"]
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    target = client.get(f"/projects/{pid}/results").json()["arterial_inflow"]
    # recover the permeability that reproduces that inflow
    r = client.post(f"/projects/{pid}/calibrate",
                    json={"target": target, "param": "permeability", "bounds": [0.5, 6.0]}).json()
    assert r["param"] == "permeability"
    assert r["value"] == pytest.approx(2.0, rel=0.15)
    assert r["achieved"] == pytest.approx(target, rel=1e-2)


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


def test_export_network_vtp(client):
    pid = client.post("/projects", json={"name": "vtp", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    # vtp exports the vessel network and needs no solved result
    r = client.get(f"/projects/{pid}/export", params={"fmt": "vtp"})
    assert r.status_code == 200
    body = r.text
    assert 'type="PolyData"' in body
    assert 'Name="radius"' in body
    assert "<Lines>" in body


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


def _drawn_tree(x0, y_root, y_tip, inlet=0):
    """A small Y-shaped tree drawn node by node, as the canvas editor sends it."""
    return {"node_x": [x0, x0, x0 - 20, x0 + 20], "node_y": [y_root, 60, y_tip, y_tip],
            "seg_start": [0, 1, 1], "seg_end": [1, 2, 3], "seg_radius": [3, 2, 2],
            "inlet_nodes": [inlet], "root_node": inlet}


def test_editable_creates_tree_from_scratch_and_solves(client):
    """The canvas editor draws vessels straight into the network: PUT /editable
    creates trees the project didn't have, with inlet/outlet *nodes*, and the run
    feeds exactly those nodes (no I/O points needed)."""
    pid = client.post("/projects", json={"name": "drawn", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    body = {"artery": _drawn_tree(-20, 10, 120), "vein": _drawn_tree(20, 150, 30)}
    r = client.put(f"/projects/{pid}/editable", json=body)
    assert r.status_code == 200, r.text
    ed = client.get(f"/projects/{pid}/editable").json()
    assert ed["artery"]["inlet_nodes"] == [0]
    assert ed["vein"]["inlet_nodes"] == [0]
    assert len(ed["artery"]["seg_start"]) == 3

    job = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()
    final = _wait_for_job(client, job["id"])
    assert final["state"] == "done", final.get("error")
    res = client.get(f"/projects/{pid}/results").json()
    assert res["arterial_segments"] == 3
    assert res["venous_segments"] == 3
    assert res["arterial_inflow"] > 0
    assert res["arterial_inflow"] == pytest.approx(res["venous_outflow"], rel=1e-6)


def test_editable_keeps_multiple_outlets_across_reload(client, tmp_path):
    """Several inlet/outlet nodes must survive the store's .npz round trip."""
    pid = client.post("/projects", json={"name": "drawn", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    vein = _drawn_tree(20, 150, 30)
    vein["inlet_nodes"] = [0, 2, 3]
    assert client.put(f"/projects/{pid}/editable", json={"vein": vein}).status_code == 200
    fresh = TestClient(create_app(store=ProjectStore(tmp_path)))     # reopen from disk
    assert fresh.get(f"/projects/{pid}/editable").json()["vein"]["inlet_nodes"] == [0, 2, 3]


def test_run_rejects_tree_whose_every_end_is_an_outlet(client):
    """A vein with no free (terminal) end can't drain the tissue -> clear error,
    not a silent zero-flow 'solution'."""
    pid = client.post("/projects", json={"name": "drawn", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    vein = {"node_x": [20, 20, 20], "node_y": [150, 90, 30], "seg_start": [0, 1],
            "seg_end": [1, 2], "seg_radius": [2, 2], "inlet_nodes": [0, 2]}
    client.put(f"/projects/{pid}/editable",
               json={"artery": _drawn_tree(-20, 10, 120), "vein": vein})
    job = client.post(f"/projects/{pid}/run", json={"use_vessels": True}).json()
    final = _wait_for_job(client, job["id"])
    assert final["state"] == "error"
    assert "venous tree has no free ends" in final["error"]


def test_prune_removes_terminal_generations(client):
    pid = client.post("/projects", json={"name": "prune", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    client.put(f"/projects/{pid}/editable", json={"artery": _drawn_tree(0, 10, 120)})
    r = client.post(f"/projects/{pid}/prune", json={"target": "artery", "steps": 1})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["n_removed"] == 2          # both branches of the Y; the trunk stays
    assert info["n_segments"] == 1
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert len(a["seg_start"]) == 1
    assert a["inlet_nodes"] == [0]
    # the default fractal tree is pruned when nothing was drawn
    r = client.post(f"/projects/{pid}/prune", json={"target": "vein", "steps": 2}).json()
    assert r["n_removed"] > 0
    assert client.post(f"/projects/{pid}/prune", json={"target": "x"}).status_code == 422


def test_editable_new_tree_defaults_inlet_to_root(client):
    pid = client.post("/projects", json={"name": "drawn", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    tree = _drawn_tree(0, 10, 120)
    tree.pop("inlet_nodes")
    tree["root_node"] = 2
    assert client.put(f"/projects/{pid}/editable", json={"artery": tree}).status_code == 200
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert a["inlet_nodes"] == [2]


def test_editable_zero_segments_removes_tree(client):
    pid = client.post("/projects", json={"name": "drawn", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    client.put(f"/projects/{pid}/editable", json={"artery": _drawn_tree(0, 10, 120)})
    empty = {"node_x": [], "node_y": [], "seg_radius": [], "seg_start": [], "seg_end": []}
    assert client.put(f"/projects/{pid}/editable", json={"artery": empty}).status_code == 200
    assert client.get(f"/projects/{pid}/editable").json()["artery"] is None
    # clearing a tree that doesn't exist is a harmless no-op
    assert client.put(f"/projects/{pid}/editable", json={"vein": empty}).status_code == 200


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


@pytest.mark.parametrize(("dim", "shape", "fill"), [
    (2, "ellipse", "colonization"), (2, "kidney", "cco"), (2, "wedge", "none"),
    (3, "ellipsoid", "colonization"), (3, "cylinder", "cco"),
])
def test_explore_grow_shapes_fills_and_dims(client, dim, shape, fill):
    body = {"dim": dim, "shape": shape, "fill": fill, "generations": 3,
            "n_points": 200, "n_terminals": 20, "nx": 20 if dim == 3 else 40, "ny": 46}
    r = client.post("/explore/grow", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["dim"] == dim
    assert d["stats"]["artery"]["segments"] >= 7          # 3 fractal generations at least
    assert d["inflow"] > 0
    seg_len = 8 if dim == 3 else 6
    assert all(len(s) == seg_len for s in d["segments"])
    if dim == 2:
        assert len(d["outline"]) >= 3
        assert d["field"]["shape"] == [40, 46]
    else:
        assert d["field"]["voxels"]
        assert len(d["field"]["voxels"][0]) == 4


@pytest.mark.parametrize("field", ["pressure", "flux", "tof"])
def test_explore_grow_3d_computes_flux_tof_and_objective(client, field):
    body = {"dim": 3, "shape": "ellipsoid", "fill": "none", "generations": 3, "nx": 16,
            "field": field, "kind": "tof_std"}
    d = client.post("/explore/grow", json=body).json()
    assert d["field"]["name"] == field
    vals = [v[3] for v in d["field"]["voxels"]]
    assert all(np.isfinite(vals))
    assert d["field"]["min"] <= d["field"]["max"]
    assert d["kind"] == "tof_std"
    assert np.isfinite(d["objective"])


def test_explore_grow_generations_add_branches_and_roots_move(client):
    def n_art(**kw):
        body = {"fill": "none", "solve": False, **kw}
        return client.post("/explore/grow", json=body).json()["stats"]["artery"]["segments"]
    assert n_art(generations=2) < n_art(generations=3) < n_art(generations=4)
    d = client.post("/explore/grow", json={"fill": "none", "solve": False,
                                           "artery": {"root": [-30, 20], "angle": 60}}).json()
    assert d["roots"]["artery"] == [-30, 20]
    assert d["segments"][0][:2] == [-30, 20]                 # the trunk starts at the root


def test_explore_grow_custom_polygon_and_fill_steps(client):
    poly = [[-50, 10], [50, 10], [40, 150], [-40, 150]]
    base = {"shape": "custom", "polygon": poly, "fill": "colonization", "solve": False}
    few = client.post("/explore/grow", json={**base, "fill_steps": 3}).json()
    many = client.post("/explore/grow", json={**base, "fill_steps": 30}).json()
    assert few["outline"] == poly
    assert few["stats"]["artery"]["segments"] < many["stats"]["artery"]["segments"]
    bad = client.post("/explore/grow", json={"shape": "custom", "polygon": [[0, 0], [1, 1]]})
    assert bad.status_code == 422


def test_shapes_endpoints(client):
    s = client.get("/shapes").json()
    assert {"heart", "kidney", "brain", "liver", "lung"} <= set(s["2d"])
    assert {"heart", "kidney", "brain", "liver", "lung"} <= set(s["3d"])
    r = client.post("/shapes/polygon", json={"name": "heart", "domain": [0, 100, 0, 120]})
    assert r.status_code == 200
    poly = np.array(r.json()["polygon"])
    assert poly[:, 0].min() >= 0
    assert poly[:, 0].max() <= 100
    assert client.post("/shapes/polygon", json={"name": "nope"}).status_code == 422
    assert client.post("/shapes/polygon",
                       json={"name": "heart", "domain": [5, 1, 0, 1]}).status_code == 422


def test_explore_grow_extent_and_organs(client):
    d = client.post("/explore/grow", json={"shape": "brain", "extent": 2.0, "fill": "none",
                                           "solve": False}).json()
    assert d["domain"] == [-140.0, 140.0, 0.0, 320.0]
    # the brain's anatomical arterial entry (circle of Willis, below), scaled ×2
    from perfusion import organs
    anat = organs.roots("brain", 2, d["domain"])
    assert d["roots"]["artery"] == pytest.approx(anat["artery"]["root"])
    assert d["aims"]["artery"] == pytest.approx(90.0)
    d3 = client.post("/explore/grow", json={"dim": 3, "shape": "heart", "nx": 14,
                                            "twist": 70, "jitter": 0.5, "fill": "none",
                                            "generations": 3}).json()
    assert d3["inflow"] > 0
    zs = [s[2] for s in d3["segments"]] + [s[5] for s in d3["segments"]]
    xs = [s[0] for s in d3["segments"]] + [s[3] for s in d3["segments"]]
    assert np.ptp(xs) > 1                                   # not confined to a plane
    assert np.ptp(zs) > 1


def test_explore_grow_large_domain_with_scaled_trees(client):
    """The explorer opens at ×4 and can scale to ×16: scaled roots/calibre are valid."""
    for e in (4, 16):
        body = {"extent": e, "fill": "colonization", "nx": 50, "ny": 57,
                "artery": {"root": [0, 8 * e], "radius": 2 * e},
                "vein": {"root": [0, 152 * e], "angle": -90, "radius": 2.4 * e}}
        r = client.post("/explore/grow", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["domain"] == [-70 * e, 70 * e, 0, 160 * e]
        assert d["inflow"] > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_explore_grow_places_roots_where_the_organ_attaches(client, dim):
    """Picking an organ puts the artery/vein where they attach anatomically, aimed
    into the tissue; a root the user places is respected."""
    from perfusion import organs
    shapes = ["kidney", "heart", "liver", "lung", "brain"]
    roots = {}
    for shape in shapes:
        body = {"dim": dim, "shape": shape, "fill": "none", "solve": False, "nx": 16}
        d = client.post("/explore/grow", json=body).json()
        anat = organs.roots(shape, dim, d["domain"])
        assert d["roots"]["artery"] == pytest.approx(anat["artery"]["root"])
        assert d["roots"]["vein"] == pytest.approx(anat["vein"]["root"])
        assert d["segments"][0][:dim] == pytest.approx(anat["artery"]["root"], abs=0.01)
        roots[shape] = tuple(np.round(d["roots"]["artery"], 3))
    assert len(set(roots.values())) == len(shapes)          # each organ differs
    # kidney: artery and vein both at the hilum, pointing into the organ (+x)
    k = client.post("/explore/grow", json={"dim": dim, "shape": "kidney", "fill": "none",
                                           "solve": False, "nx": 16}).json()
    if dim == 2:
        assert k["aims"]["artery"] == pytest.approx(0.0)
    else:
        assert k["aims"]["artery"] == pytest.approx([1.0, 0.0, 0.0])
    # a placed root wins over the anatomical default
    mine = [1.0, 2.0] if dim == 2 else [10.0, 11.0, 12.0]
    d = client.post("/explore/grow", json={"dim": dim, "shape": "kidney", "fill": "none",
                                           "solve": False, "nx": 16,
                                           "artery": {"root": mine}}).json()
    assert d["roots"]["artery"] == mine


def test_default_trees_are_fitted_into_the_project_domain(client):
    """A project with no vessels runs on default fractal trees — fitted inside its own
    domain (the raw thesis trees sit in a fixed frame, partly outside most domains),
    and reported by /network so the UI can draw what is solved."""
    for dom in ([0, 100, 0, 120], [0, 400, 0, 200]):
        grid = {"nx": 40, "ny": 40, "lx": dom[1] - dom[0], "ly": dom[3] - dom[2],
                "domain_limit": dom}
        pid = client.post("/projects", json={"name": "d", "grid": grid,
                                             "params": FAST_PARAMS}).json()["id"]
        net = client.get(f"/projects/{pid}/network").json()
        assert sorted(net["defaults"]) == ["artery", "vein"]
        xs = [v for sg in net["segments"] for v in (sg["x0"], sg["x1"])]
        ys = [v for sg in net["segments"] for v in (sg["y0"], sg["y1"])]
        assert min(xs) >= dom[0]
        assert max(xs) <= dom[1]
        assert min(ys) >= dom[2]
        assert max(ys) <= dom[3]
        assert max(xs) - min(xs) > 0.5 * (dom[1] - dom[0]) or \
            max(ys) - min(ys) > 0.5 * (dom[3] - dom[2])      # actually fills the domain
        jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
        assert _wait_for_job(client, jid)["state"] == "done"
        res = client.get(f"/projects/{pid}/results").json()
        assert res["arterial_inflow"] > 0
        assert res["arterial_inflow"] == pytest.approx(res["venous_outflow"], rel=1e-6)


def test_uq_with_default_trees_still_responds_to_tree_depth(client):
    """With no drawn vessels the default trees are rebuilt per sample, so the tree
    depth (r_min_frac) remains an effective uncertain input."""
    pid = client.post("/projects", json={"name": "uq", "grid": SMALL_GRID,
                                         "params": FAST_PARAMS}).json()["id"]
    r = client.post(f"/projects/{pid}/uq", json={"params": {"r_min_frac": [0.4, 0.7]},
                                                 "n_samples": 6})
    assert r.status_code == 200, r.text
    assert r.json()["std"]["arterial_inflow"] > 0


# ── 3-D projects ───────────────────────────────────────────────────────────────
def _explorer_project(client, dim, **kw):
    body = {"dim": dim, "shape": "heart" if dim == 3 else "kidney", "fill": "cco",
            "n_terminals": 12, "generations": 3, "nx": 16 if dim == 3 else 40, "ny": 46,
            "name": f"from-explorer-{dim}d", **kw}
    r = client.post("/explore/project", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_explorer_3d_result_becomes_a_runnable_3d_project(client):
    made = _explorer_project(client, 3)
    pid = made["id"]
    assert made["open"] == f"/viewer3d.html?project={pid}"
    info = client.get(f"/projects/{pid}").json()
    assert info["grid"]["nz"] == 16
    assert info["grid"]["tissue"] == "heart"
    assert len(info["grid"]["domain_limit"]) == 6
    pre = client.get(f"/projects/{pid}/view3d").json()
    assert pre["dim"] == 3
    assert pre["tissue_points"]
    assert all(len(sg) == 8 for sg in pre["segments"])
    assert not pre["has_result"]

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    res = client.get(f"/projects/{pid}/results").json()
    assert res["arterial_inflow"] > 0
    assert res["arterial_inflow"] == pytest.approx(res["venous_outflow"], rel=1e-6)
    for field in ("pressure", "flux", "tof"):
        v = client.get(f"/projects/{pid}/view3d?field={field}").json()
        assert v["pressure"]["name"] == field
        assert v["pressure"]["voxels"]
        assert all(len(x) == 4 for x in v["pressure"]["voxels"][:5])
    assert client.get(f"/projects/{pid}/export?fmt=npz").status_code == 200


def test_3d_project_field_exports(client):
    """A 3-D result exports its voxel fields as CSV (x,y,z + fields per active voxel)
    and VTK ImageData (a voxel volume), and as one zip with the tracer frames; a
    direct-connection result has no field to export (but its vessels do)."""
    import io as stdio
    import xml.etree.ElementTree as ET
    import zipfile

    pid = _explorer_project(client, 3)["id"]
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    shape = client.get(f"/projects/{pid}/view3d").json()["shape"]

    r = client.get(f"/projects/{pid}/export?fmt=csv")
    assert r.status_code == 200, r.text
    lines = r.text.splitlines()
    assert lines[0] == "x,y,z,pressure,flux,tof"
    rows = np.loadtxt(stdio.StringIO(r.text), delimiter=",", skiprows=1)
    with np.load(stdio.BytesIO(client.get(f"/projects/{pid}/export?fmt=npz").content)) as d:
        mask, dl = d["mask"], d["domain_limit"]
    assert len(rows) == int(mask.sum())                         # one row per active voxel
    for k in range(3):                                          # centres inside the box
        assert dl[2 * k] < rows[:, k].min()
        assert rows[:, k].max() < dl[2 * k + 1]

    r = client.get(f"/projects/{pid}/export?fmt=vtk")
    assert r.status_code == 200
    img = ET.fromstring(r.text).find("ImageData")
    assert img.attrib["WholeExtent"] == " ".join(f"0 {n}" for n in shape)
    p = next(da for da in ET.fromstring(r.text).iter("DataArray")
             if da.attrib["Name"] == "pressure")
    assert len(p.text.split()) == int(np.prod(shape))

    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 8}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    r = client.get(f"/projects/{pid}/bundle")
    assert r.status_code == 200, r.text
    names = set(zipfile.ZipFile(stdio.BytesIO(r.content)).namelist())
    assert {"results.npz", "fields.csv", "fields.vti", "summary.json", "tracer.npz"} <= names
    import importlib.util

    if importlib.util.find_spec("imageio"):                     # 3-D flow video renders too
        assert "tracer.gif" in names

    jid = client.post(f"/projects/{pid}/run", json={"model": "direct"}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    for path in ("export?fmt=csv", "export?fmt=npz", "bundle"):
        r = client.get(f"/projects/{pid}/{path}")
        assert r.status_code == 409, path
        assert "no tissue field" in r.json()["detail"]
    assert client.get(f"/projects/{pid}/export?fmt=vtp").status_code == 200


def test_2d_only_features_say_so_on_a_3d_project(client):
    pid = _explorer_project(client, 3)["id"]
    for method, path, body in [
        ("put", "mask", {"regions": []}), ("get", "results/field/pressure", None),
        ("get", "tracer", None),
    ]:
        r = getattr(client, method)(f"/projects/{pid}/{path}",
                                    **({"json": body} if body is not None else {}))
        assert r.status_code == 409, (path, r.status_code, r.text)
        assert "3-D project" in r.json()["detail"]


def test_3d_project_direct_connection_model(client):
    """The direct-connection model runs on a 3-D project: 3-D links, mass balance, the
    viewer scene (trees + links by flow) and the network-timed bolus — and the hybrid
    run afterwards replaces it."""
    pid = _explorer_project(client, 3)["id"]
    jid = client.post(f"/projects/{pid}/run", json={"model": "direct", "rho_c": 0}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    summary = client.get(f"/projects/{pid}/results").json()
    assert summary["model"] == "direct"
    assert summary["arterial_inflow"] > 0
    assert summary["n_connections"] > 0

    d = client.get(f"/projects/{pid}/direct").json()
    assert d["dim"] == 3
    assert len(d["artery"][0]) == len(d["connections"][0]) == 7     # [x,y,z]×2 + flow
    assert len(d["connections"]) == summary["n_connections"]
    assert sum(c[6] for c in d["connections"]) == pytest.approx(summary["arterial_inflow"])

    v = client.get(f"/projects/{pid}/view3d").json()
    assert v["direct"]["dim"] == 3
    assert v["summary"]["model"] == "direct"
    assert not v["pressure"]["voxels"]
    r = client.get(f"/projects/{pid}/view3d?field=tof")
    assert r.status_code == 409
    assert "no tissue field" in r.json()["detail"]
    assert client.get(f"/projects/{pid}/view3d?field=tracer").status_code == 409

    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 10}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    t = client.get(f"/projects/{pid}/view3d?field=tracer").json()["net_tracer"]
    assert len(t["lines"][0]) == 6
    assert len(t["frames"]) == 10
    assert len(t["frames"][0]) == len(t["lines"]) == len(t["kind"])
    assert 2 in t["kind"]                                            # connections animate

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]   # back to the hybrid
    assert _wait_for_job(client, jid)["state"] == "done"
    v = client.get(f"/projects/{pid}/view3d").json()
    assert "direct" not in v
    assert v["pressure"]["voxels"]
    assert client.get(f"/projects/{pid}/direct").status_code == 409


def test_3d_project_flow_kinetics_and_oxygen(client):
    """Tracer, kinetics, kinetic maps and oxygen run on a 3-D project and show in
    the 3-D viewer (voxel fields + a per-frame tracer animation)."""
    pid = _explorer_project(client, 3)["id"]
    assert client.get(f"/projects/{pid}/view3d?field=tracer").status_code == 409  # no tracer
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    jid = client.post(f"/projects/{pid}/tracer", json={"n_frames": 12}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"

    kin = client.get(f"/projects/{pid}/kinetics?model=1CM").json()
    assert kin["Fp"] > 0
    assert len(kin["measured"]) == 12
    fp = client.get(f"/projects/{pid}/kinetics/maps?field=Fp").json()
    assert len(fp["shape"]) == 3
    ox = client.get(f"/projects/{pid}/oxygen?field=saturation&k=1").json()
    assert len(ox["shape"]) == 3
    # the uptake rate is scaled by the *organ* volume, not the bounding box, so the
    # well-mixed estimate (~1 - e^-k, lower with heterogeneity) holds inside a mask
    assert 0.25 < ox["oef"] < 0.75

    anim = client.get(f"/projects/{pid}/view3d?field=tracer").json()["tracer"]
    assert len(anim["frames"]) == 12
    assert len(anim["frames"][0]) == len(anim["xyz"])
    assert max(max(f) for f in anim["frames"]) > 0
    for field in ("saturation", "Fp", "vp", "CBV", "MTT"):
        v = client.get(f"/projects/{pid}/view3d?field={field}").json()
        assert v["pressure"]["name"] == field
        assert v["pressure"]["voxels"], field
    assert 0 < client.get(f"/projects/{pid}/view3d?field=saturation&k=2").json()["oef"] < 1
    assert client.get(f"/projects/{pid}/view3d?field=nope").status_code == 422


def test_3d_project_uncertainty_and_calibration(client):
    """UQ (Pearson + Sobol) and calibration re-solve a 3-D project's own model."""
    pid = _explorer_project(client, 3)["id"]
    r = client.post(f"/projects/{pid}/uq", json={"params": {"p_in": [5, 15], "mu": [0.8, 1.5]},
                                                 "n_samples": 6})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mean"]["arterial_inflow"] > 0
    assert d["sensitivity"]["arterial_inflow"]["p_in"] > 0.5
    r = client.post(f"/projects/{pid}/uq", json={"params": {"p_in": [5, 15],
                                                            "permeability": [0.5, 2]},
                                                 "n_samples": 10, "method": "sobol"})
    assert r.status_code == 200, r.text
    assert r.json()["fit_r2"]["arterial_inflow"] > 0.9

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    base = client.get(f"/projects/{pid}/results").json()["arterial_inflow"]
    r = client.post(f"/projects/{pid}/calibrate",
                    json={"target": 1.5 * base, "param": "p_in", "bounds": [1, 40]})
    assert r.status_code == 200, r.text
    cal = r.json()
    assert cal["value"] == pytest.approx(1.5 * 10.0, rel=1e-3)   # inflow is linear in p_in
    assert cal["achieved"] == pytest.approx(1.5 * base, rel=1e-5)

    # tree depth / hematocrit don't enter the 3-D solve → a clear 400, not a silent no-op
    r = client.post(f"/projects/{pid}/uq",
                    json={"params": {"r_min_frac": [0.3, 0.6]}, "n_samples": 4})
    assert r.status_code == 400
    assert "3-D project" in r.json()["detail"]
    r = client.post(f"/projects/{pid}/calibrate",
                    json={"target": 1.0, "param": "hematocrit", "bounds": [0.2, 0.6]})
    assert r.status_code == 400


def test_3d_project_multicompartment_bed(client):
    pid = _explorer_project(client, 3)["id"]
    perf = client.get(f"/projects/{pid}/multicompartment", params={"n": 3, "beta": 1.0}).json()
    assert perf["shape"] == [16, 16, 16]
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    inflow = client.get(f"/projects/{pid}/results").json()["arterial_inflow"]
    v = client.get(f"/projects/{pid}/view3d", params={"field": "mc_perfusion", "n": 3}).json()
    assert v["pressure"]["voxels"]
    assert v["mc"]["n"] == 3
    # every unit of blood crosses the arteriole→capillary interface (≈ the 1-bed inflow)
    assert v["mc"]["perfusion"] == pytest.approx(inflow, rel=0.2)
    a = client.get(f"/projects/{pid}/view3d", params={"field": "mc_arteriole"}).json()
    vn = client.get(f"/projects/{pid}/view3d", params={"field": "mc_venule"}).json()

    def mean(view):
        return np.mean([x[3] for x in view["pressure"]["voxels"]])
    assert mean(a) > mean(vn)                            # arteriole → venule cascade


def test_3d_project_trees_are_hand_editable(client):
    """PUT /editable on a 3-D project: add a branch, move a node in z, resize, toggle an
    inlet — z survives the rebuild, lengths are 3-D, and the edited trees solve."""
    pid = _explorer_project(client, 3)["id"]
    a = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert a["node_z"] is not None
    n0, s0 = len(a["node_x"]), len(a["seg_start"])
    tip = a["seg_end"][-1]
    a["node_x"].append(a["node_x"][tip] + 3)
    a["node_y"].append(a["node_y"][tip])
    a["node_z"].append(a["node_z"][tip] + 4)
    a["seg_start"].append(tip)
    a["seg_end"].append(n0)
    a["seg_radius"].append(0.5)
    a["node_z"][1] += 1.5
    a["seg_radius"][0] *= 1.3
    a["inlet_nodes"] = [a["root_node"], 3]
    keys = ("node_x", "node_y", "node_z", "seg_start", "seg_end", "seg_radius",
            "inlet_nodes", "root_node")
    r = client.put(f"/projects/{pid}/editable", json={"artery": {k: a[k] for k in keys}})
    assert r.status_code == 200, r.text
    b = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert len(b["node_x"]) == n0 + 1
    assert len(b["seg_start"]) == s0 + 1
    assert b["seg_length"][-1] == pytest.approx(5.0)               # √(3² + 4²): 3-D length
    assert b["node_z"][1] == pytest.approx(a["node_z"][1])
    assert b["seg_radius"][0] == pytest.approx(a["seg_radius"][0])
    assert sorted(b["inlet_nodes"]) == [a["root_node"], 3]

    # a pure move keeps the topology and refreshes the 3-D lengths
    mv = {"node_x": b["node_x"], "node_y": b["node_y"],
          "node_z": [z + (2.0 if i == n0 else 0.0) for i, z in enumerate(b["node_z"])],
          "seg_radius": b["seg_radius"]}
    assert client.put(f"/projects/{pid}/editable", json={"artery": mv}).status_code == 200
    c = client.get(f"/projects/{pid}/editable").json()["artery"]
    assert c["seg_length"][-1] == pytest.approx(np.sqrt(9 + 36))

    no_z = {k: a[k] for k in keys if k != "node_z"}
    r = client.put(f"/projects/{pid}/editable", json={"artery": no_z})
    assert r.status_code == 422
    assert "node_z" in r.json()["detail"]

    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"
    res = client.get(f"/projects/{pid}/results").json()
    assert res["arterial_inflow"] == pytest.approx(res["venous_outflow"], rel=1e-6)


def test_2d_project_edits_reject_node_z(client):
    pid = client.post("/projects", json={"name": "e2", "grid": SMALL_GRID}).json()["id"]
    body = {"node_x": [0, 10], "node_y": [0, 10], "node_z": [0, 1], "seg_start": [0],
            "seg_end": [1], "seg_radius": [1.0]}
    r = client.put(f"/projects/{pid}/editable", json={"artery": body})
    assert r.status_code == 422
    assert "only for 3-D" in r.json()["detail"]


@pytest.mark.parametrize("method", ["colonization", "cco"])
def test_3d_project_grows_into_its_tissue(client, method):
    pid = _explorer_project(client, 3)["id"]
    before = client.get(f"/projects/{pid}/view3d").json()["segments"]
    r = client.post(f"/projects/{pid}/grow",
                    json={"target": "artery", "method": method, "n_points": 40})
    assert r.status_code == 200, r.text
    after = client.get(f"/projects/{pid}/view3d").json()
    assert len(after["segments"]) > len(before)
    zs = [z for sg in after["segments"] for z in (sg[2], sg[5])]
    assert max(zs) - min(zs) > 0                       # still a volumetric tree
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_3d_project_adapts_toward_murray(client):
    pid = _explorer_project(client, 3)["id"]
    r = client.post(f"/projects/{pid}/adapt", json={"n_iter": 20})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["n_iter"] >= 1
    assert abs(info["murray_final"] - 3) <= abs(info["murray_start"] - 3)
    assert client.get(f"/projects/{pid}/document").json()["artery_network"]["node_z"]


def test_3d_project_prune_morphometry_and_document_round_trip(client):
    pid = _explorer_project(client, 3)["id"]
    before = client.get(f"/projects/{pid}/morphometry").json()
    assert before["max_order"] >= 2
    r = client.post(f"/projects/{pid}/prune", json={"target": "artery", "steps": 1})
    assert r.status_code == 200
    assert r.json()["n_removed"] > 0
    doc = client.get(f"/projects/{pid}/document").json()
    assert doc["grid"]["nz"] == 16
    assert len(doc["mask"]["shape"]) == 3
    assert doc["artery_network"]["node_z"] is not None
    new = client.post("/projects/import", json=doc).json()["id"]
    jid = client.post(f"/projects/{new}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_explorer_2d_result_becomes_an_editable_workspace_project(client):
    made = _explorer_project(client, 2)
    pid = made["id"]
    assert made["open"] == f"/?project={pid}"
    assert len(client.get(f"/projects/{pid}/mask").json()["regions"]) == 1
    ed = client.get(f"/projects/{pid}/editable").json()
    assert ed["artery"] is not None
    assert ed["vein"] is not None
    jid = client.post(f"/projects/{pid}/run", json={}).json()["id"]
    assert _wait_for_job(client, jid)["state"] == "done"


def test_grid_dims_are_validated(client):
    bad = {"name": "x", "grid": {"nx": 10, "ny": 10, "nz": 10, "domain_limit": [0, 1, 0, 1]}}
    assert client.post("/projects", json=bad).status_code == 422


def test_tracer_images_3d_projects_like_the_viewer():
    """The 3-D video renderer: the domain centre lands mid-frame, a single hot voxel
    lights its projected pixel (maximum-intensity projection), vessels are drawn,
    the time bar grows, and ``spin`` turns the camera between frames."""
    pytest.importorskip("matplotlib")
    from perfusion.api import helpers

    col, row, _ = helpers.project_3d(np.array([[5.0, 5.0, 5.0]]), np.array([5.0, 5.0, 5.0]),
                                     10.0, 0.7, 0.45, 200)
    assert (col[0], row[0]) == (100.0, 100.0)

    mask = np.ones((5, 5, 5), bool)
    frames = np.zeros((3, 5, 5, 5))
    frames[1, 2, 2, 2] = 1.0                                    # centre voxel, frame 1
    tracer = {"frames": frames, "mask": mask, "vmax": 1.0}
    seg = [[0, 0, 0, 10, 10, 10, 1.0, 0]]                       # one artery diagonal
    imgs = helpers.tracer_images_3d(tracer, (0, 10, 0, 10, 0, 10), seg, size=120, spin=0.0)
    assert len(imgs) == 3
    assert imgs[0].shape == (120, 120, 3)
    assert imgs[0].dtype == np.uint8
    centre = imgs[1][55:65, 55:65].astype(int)
    assert (centre[..., 1] > 150).any()                         # yellow-green bolus
    assert not (imgs[0][55:65, 55:65, 1] > 150).any()           # frame 0: no contrast
    red = (imgs[0][..., 0] > 150) & (imgs[0][..., 2] < 140)
    assert red.any()                                            # the artery is drawn
    bars = [int((im[-1, :, 0] == 255).sum()) for im in imgs]
    assert bars == sorted(bars)
    assert bars[-1] == 120
    turning = helpers.tracer_images_3d(tracer, (0, 10, 0, 10, 0, 10), seg, size=120)
    assert not np.array_equal(turning[2], imgs[2])              # spin moves the camera
