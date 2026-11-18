"""The Tree explorer: anatomical roots per organ, manual roots, 3-D fields."""

from pathlib import Path

import numpy as np
import pytest

from .conftest import wait_status


def _ready(page):
    page.wait_for_function("document.querySelector('#status').textContent === 'ready'",
                           timeout=120000)


def test_organ_change_places_anatomical_roots(page, server):
    from perfusion import organs

    page.goto(server + "/explorer.html")
    _ready(page)
    for organ in ("kidney", "liver", "brain"):
        page.select_option("#shape2", organ)
        page.wait_for_timeout(250)
        _ready(page)
        ext = float(page.input_value("#extent"))
        anat = organs.roots(organ, 2, (-70 * ext, 70 * ext, 0.0, 160 * ext))
        assert float(page.input_value("#a-x")) == pytest.approx(anat["artery"]["root"][0], abs=0.1)
        assert float(page.input_value("#a-y")) == pytest.approx(anat["artery"]["root"][1], abs=0.1)
    # a manually placed root sticks; Organ default restores the anatomy
    page.fill("#a-x", "-100")
    page.dispatch_event("#a-x", "change")
    _ready(page)
    assert "placed by you" in page.inner_text("#root-mode")
    page.click("#btn-root-default")
    _ready(page)
    assert "organ default" in page.inner_text("#root-mode")
    assert not page.errors


def test_3d_tree_computes_time_of_flight_and_objective(page, server):
    page.goto(server + "/explorer.html")
    _ready(page)
    page.click("#dim-3")
    page.wait_for_timeout(300)
    _ready(page)
    page.select_option("#shape3", "heart")
    page.wait_for_timeout(200)
    _ready(page)
    page.select_option("#field", "tof")
    page.wait_for_timeout(200)
    _ready(page)
    assert page.inner_text("#cb-title") == "tof"
    assert "tof_std =" in page.inner_text("#obj-info")
    assert "PERFUSED INFLOW" in page.inner_text("#summary").upper()
    wait_status(page, "ready")
    assert not page.errors


def test_3d_explorer_result_opens_as_a_project_and_runs(page, server):
    page.goto(server + "/explorer.html")
    _ready(page)
    page.click("#dim-3")
    page.wait_for_timeout(300)
    _ready(page)
    page.select_option("#fill", "cco")
    _ready(page)
    with page.context.expect_page() as new:
        page.click("#btn-project")
    viewer = new.value
    viewer.wait_for_load_state()
    assert "/viewer3d.html?project=" in viewer.url
    viewer.wait_for_selector("#proj-card:not([hidden])")
    viewer.click("#v3-run")
    viewer.wait_for_function("document.querySelector('#v3-summary').textContent"
                             ".includes('arterial inflow')", timeout=120000)
    viewer.select_option("#v3-field", "tof")
    viewer.wait_for_function("document.querySelector('#status').textContent"
                             ".includes('solved')")
    assert "segments" in viewer.inner_text("#proj-desc")


def test_2d_explorer_result_opens_in_the_workspace(page, server):
    page.goto(server + "/explorer.html")
    _ready(page)
    page.select_option("#shape2", "liver")
    _ready(page)
    with page.context.expect_page() as new:
        page.click("#btn-project")
    ws = new.value
    ws.wait_for_load_state()
    ws.wait_for_function("document.querySelector('#status').textContent.includes('opened')",
                         timeout=60000)
    assert "artery segments" in ws.inner_text("#node-info")
    ws.keyboard.press("2")
    assert ws.inner_text("#tissue-info").startswith("1 organ")


def test_3d_project_grow_flow_kinetics_and_oxygen_in_the_viewer(page, server):
    """The 3-D viewer's project tools: grow a tree, run, play the bolus, fit
    kinetics, switch to an oxygen map — with no page errors."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": "ui-3d"}).json()["id"]
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-tree-card:not([hidden])")
    assert page.is_hidden("#v3-flow-card")             # no flow before a solve
    page.fill("#v3-npts", "20")
    page.click("#v3-grow")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('terminals')", timeout=60000)
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    page.click("#v3-flow")
    page.wait_for_function("document.querySelector('#v3-kin').textContent.includes('Fp =')",
                           timeout=120000)
    assert page.is_visible("#v3-play-row")
    assert page.input_value("#v3-field") == "tracer"
    assert page.is_hidden("#v3-k-row")
    assert page.is_visible("#v3-video-row")                      # server-rendered video
    assert page.get_attribute("#v3-mp4", "href").endswith("tracer.mp4?spin=true")
    page.uncheck("#v3-video-spin")
    assert page.get_attribute("#v3-gif", "href").endswith("tracer.gif?spin=false")
    with page.expect_download(timeout=60000) as dl:
        page.click("#v3-gif")
    assert Path(dl.value.path()).read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    page.select_option("#v3-field", "saturation")
    page.wait_for_function("document.querySelector('#v3-summary').textContent"
                           ".includes('O₂ extraction')", timeout=60000)
    assert page.is_visible("#v3-k-row")
    assert page.is_hidden("#v3-play-row")
    assert not errors


def test_3d_viewer_multicompartment_uncertainty_and_calibration(page, server):
    """The 3-D viewer's analyses: a multi-compartment field, an uncertainty run and a
    calibration — all on the 3-D project, with no page errors."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": "ui-3d-an"}).json()["id"]
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-an-card:not([hidden])")
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    assert page.is_hidden("#v3-mc-row")
    page.select_option("#v3-field", "mc_perfusion")
    page.wait_for_function("document.querySelector('#v3-summary').textContent"
                           ".includes('capillary exchange')", timeout=60000)
    assert page.is_visible("#v3-mc-row")

    page.click("#v3-an-card summary")
    page.fill("#v3-uq-n", "6")
    page.click("#v3-uq")
    page.wait_for_function("document.querySelector('#v3-uq-out').textContent"
                           ".includes('sensitivity')", timeout=120000)
    assert "arterial_inflow" in page.inner_text("#v3-uq-out")
    assert page.input_value("#v3-cal-target")          # prefilled from the solved inflow
    assert page.input_value("#v3-cal-param") == "p_in"   # the default: inflow ∝ p_in
    page.select_option("#v3-cal-param", "permeability")
    assert page.input_value("#v3-cal-hi") == "10"
    page.select_option("#v3-cal-param", "p_in")
    assert page.input_value("#v3-cal-hi") == "50"
    page.click("#v3-cal")
    page.wait_for_function("document.querySelector('#v3-cal-out').textContent"
                           ".includes('solves')", timeout=120000)
    assert page.inner_text("#v3-cal-out").startswith("p_in = 12.")   # 1.2 × the inflow
    assert not errors


def test_3d_viewer_direct_connection_model(page, server):
    """Run the direct-connection model on a 3-D project in the viewer: the field picker
    and the hybrid-only analyses hide, the summary counts the links, and the network
    bolus plays — then switching back to the hybrid restores the field."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": "ui-3d-direct"}).json()["id"]
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-an-card:not([hidden])")
    assert page.is_hidden("#v3-rhoc-row")
    page.select_option("#v3-model", "direct")
    assert page.is_visible("#v3-rhoc-row")
    page.fill("#v3-rhoc", "3")
    page.click("#v3-run")
    page.wait_for_function("document.querySelector('#v3-summary').textContent"
                           ".includes('connections')", timeout=120000)
    assert page.is_hidden("#v3-field-row")
    assert page.is_hidden("#v3-an-card")
    assert "terminal connections" in page.inner_text("#info")
    assert page.input_value("#v3-export-fmt") == "vtp"      # no tissue field to export
    assert page.is_disabled("#v3-bundle")
    page.click("#v3-flow")
    page.wait_for_selector("#v3-play-row:not([hidden])", timeout=120000)
    assert page.inner_text("#v3-time").startswith("t = ")
    page.wait_for_function("document.querySelector('#v3-kin').textContent.includes('reaches')")
    import re

    links = re.search(r"([0-9.]+)connections", page.text_content("#v3-summary")).group(1)
    assert links.isdigit()                                 # a count, not "45.000"

    page.reload()                                  # the stored model is reflected on load
    page.wait_for_selector("#v3-flow-card:not([hidden])")
    assert page.input_value("#v3-model") == "direct"
    page.select_option("#v3-model", "hybrid")
    page.click("#v3-run")
    page.wait_for_function("document.querySelector('#v3-summary').textContent"
                           ".includes('MTT')", timeout=120000)
    assert page.is_visible("#v3-field-row")
    assert page.is_visible("#v3-an-card")
    assert page.is_enabled("#v3-bundle")
    assert page.input_value("#v3-export-fmt") == "npz"      # back from the forced .vtp
    page.select_option("#v3-export-fmt", "csv")
    with page.expect_download() as dl:
        page.click("#v3-export")
    assert dl.value.suggested_filename.endswith(".csv")
    assert Path(dl.value.path()).read_text().startswith("x,y,z,pressure")
    with page.expect_download() as dl:
        page.click("#v3-png")
    assert dl.value.suggested_filename.endswith("-3d.png")
    assert not errors


def test_3d_viewer_lesions_and_hotspots(page, server):
    """Place a lesion and a hotspot sphere in the 3-D viewer: edit the lesion's radius,
    drag its centre, then solve with it — saved through the API, no page errors."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": "ui-3d-lesion"}).json()["id"]
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-reg-card:not([hidden])")
    page.uncheck("#spin")
    page.click("#v3-add-lesion")
    page.wait_for_function("document.querySelector('#v3-reg-info').textContent"
                           ".includes('lesions saved')")
    les = httpx.get(f"{server}/projects/{pid}/lesions").json()["lesions"]
    assert len(les) == 1
    assert les[0]["permeability"] == pytest.approx(0.1)
    c0 = les[0]["center"]

    page.fill('#v3-reg-list input[data-kind="lesion"][data-f="radius"]', "7")
    page.press('#v3-reg-list input[data-kind="lesion"][data-f="radius"]', "Enter")
    page.wait_for_function("document.querySelector('#v3-reg-info').textContent"
                           ".includes('lesions saved')")
    assert httpx.get(f"{server}/projects/{pid}/lesions").json()["lesions"][0]["radius"] == 7

    # drag the sphere's centre handle 60 px to the right
    sx, sy = page.evaluate("""() => {
      const r = window.viewer3d.regions.st.lesion[0];
      return window.viewer3d.project(...r.center).slice(0, 2); }""")
    box = page.locator("#view3d").bounding_box()
    page.mouse.move(box["x"] + sx, box["y"] + sy)
    page.mouse.down()
    page.mouse.move(box["x"] + sx + 30, box["y"] + sy, steps=4)
    page.mouse.move(box["x"] + sx + 60, box["y"] + sy, steps=4)
    page.mouse.up()
    page.wait_for_function(f"""() => {{
      const c = window.viewer3d.regions.st.lesion[0].center;
      return Math.hypot(c[0] - {c0[0]}, c[1] - {c0[1]}, c[2] - {c0[2]}) > 1; }}""")
    page.wait_for_timeout(500)
    moved = httpx.get(f"{server}/projects/{pid}/lesions").json()["lesions"][0]["center"]
    assert np.linalg.norm(np.subtract(moved, c0)) > 1

    page.click("#v3-add-hotspot")
    page.wait_for_function("document.querySelector('#v3-reg-info').textContent"
                           ".includes('hotspots saved')")
    assert len(httpx.get(f"{server}/projects/{pid}/hotspots").json()["hotspots"]) == 1
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    assert page.locator(".reg-row").count() == 2
    assert not errors


def _3d_project(server, name):
    import httpx

    return httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": name}).json()["id"]


def test_3d_viewer_compares_runs(page, server):
    """In the 3-D viewer: keep a baseline, add a lesion, re-run, compare and show the
    signed Δ-flux field."""
    pid = _3d_project(server, "ui-3d-compare")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-cmp-card:not([hidden])")
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    page.click("#v3-cmp-keep")
    page.wait_for_function("document.querySelector('#v3-cmp-info').textContent.includes('kept')")
    page.click("#v3-add-lesion")
    page.wait_for_function("document.querySelector('#v3-reg-info').textContent"
                           ".includes('lesions saved')")
    page.click("#v3-run")
    page.wait_for_function("document.querySelector('#v3-run').textContent.includes('Re-run')"
                           " && !document.querySelector('#v3-run').disabled", timeout=120000)
    page.click("#v3-cmp-run")
    page.wait_for_selector("#v3-cmp-out table.cmp")
    assert "run 1" in page.inner_text("#v3-cmp-out")
    page.click('#v3-cmp-maps [data-delta="flux"]')
    page.wait_for_function("window.viewer3d && document.querySelector('#v3-field').value"
                           " === 'd_flux' && document.querySelector('#status').textContent"
                           ".includes('solved')", timeout=60000)
    assert page.evaluate("window.viewer3d.data().delta") is True
    assert page.locator("#v3-cmp-bparam option[value=r_min_frac]").count() == 0   # 2-D only
    page.select_option("#v3-cmp-bparam", "permeability")
    page.fill("#v3-cmp-bvals", "0.5, 2")
    page.click("#v3-cmp-brun")
    page.wait_for_function("document.querySelector('#v3-cmp-info').textContent"
                           ".includes('runs kept')", timeout=120000)
    assert page.locator("#v3-cmp-bout tr[data-snap]").count() == 2
    assert not errors


def test_3d_viewer_fits_a_dce_curve(page, server):
    """3-D viewer: compute the flow, use its curve as data and fit p_in + porosity."""
    pid = _3d_project(server, "ui-3d-dce")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-an-card:not([hidden])")
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    page.click("#v3-flow")
    page.wait_for_selector("#v3-play-row:not([hidden])", timeout=120000)
    page.click("#v3-an-card summary")
    page.click("#v3-dce-own")
    page.wait_for_function("document.querySelector('#v3-dce-csv').value.split('\\n').length > 10")
    page.click("#v3-dce-go")
    page.wait_for_selector("#v3-dce-out table.cmp", timeout=180000)
    assert "porosity" in page.inner_text("#v3-dce-out")
    assert page.is_visible("#v3-dce-chart")
    assert not errors


def test_3d_viewer_loops_and_occlusion(page, server):
    """3-D viewer: ⟳ Loops adds anastomoses; ⛔ Occlude narrows the selected vessel."""
    import httpx

    pid = _3d_project(server, "ui-3d-loops")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-tree-card:not([hidden])")
    page.click("#v3-loops")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('anastomoses')", timeout=60000)
    page.check("#v3-edit")
    page.wait_for_function("window.viewer3d.editor.active")
    r0 = httpx.get(f"{server}/projects/{pid}/editable").json()["artery"]["seg_radius"][2]
    page.evaluate("""() => { const ed = window.viewer3d.editor;
      ed.sel = { kind: 'artery', type: 'seg', i: 2 }; ed.refreshPanel(); }""")
    page.click("#v3-occlude")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('radius')", timeout=60000)
    r1 = httpx.get(f"{server}/projects/{pid}/editable").json()["artery"]["seg_radius"][2]
    assert r1 == pytest.approx(max(1e-3, r0 * 1e-3), rel=1e-6)
    assert not errors


def test_3d_viewer_hand_edits_the_trees(page, server):
    """Edit a 3-D project's trees in the viewer: select + resize a vessel, drag a
    node, branch with Shift+click, toggle an outlet, delete, undo — then solve."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "none", "generations": 3,
        "nx": 16, "name": "ui-3d-edit"}).json()["id"]

    def tree(kind):
        return httpx.get(f"{server}/projects/{pid}/editable").json()[kind]

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-tree-card:not([hidden])")
    page.check("#v3-edit")
    page.wait_for_selector("#v3-edit-panel:not([hidden])")
    box = page.locator("#view3d").bounding_box()

    def at(kind, i):
        x, y = page.evaluate(f"window.viewer3d.editor.screenOf('{kind}', {i})")
        return box["x"] + x, box["y"] + y

    def saved(n_before, key="seg_start", kind="artery", delta=1):
        page.wait_for_function("!document.querySelector('#v3-tree-info').textContent"
                               ".includes('…')")
        for _ in range(50):
            if len(tree(kind)[key]) == n_before + delta:
                return
            page.wait_for_timeout(100)
        raise AssertionError(f"{kind} {key} did not change by {delta}")

    a0 = tree("artery")
    # select the trunk's tip node and drag it
    tip = a0["seg_end"][0]
    x, y = at("artery", tip)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 25, y - 15, steps=5)
    page.mouse.up()
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('node moved')", timeout=30000)
    a1 = tree("artery")
    moved = np.hypot(np.subtract(a1["node_x"], a0["node_x"]),
                     np.subtract(a1["node_z"], a0["node_z"]))
    assert moved.max() > 0.5                             # the node moved in 3-D
    # Shift+click empty space: a new branch from the selected node
    page.keyboard.down("Shift")
    page.mouse.click(x + 70, y + 40)
    page.keyboard.up("Shift")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('branch added')", timeout=30000)
    assert len(tree("artery")["seg_start"]) == len(a0["seg_start"]) + 1
    # select the new vessel (between the two new nodes) and resize it
    a2 = tree("artery")
    s_new = len(a2["seg_start"]) - 1
    (ax, ay), (bx, by) = at("artery", a2["seg_start"][s_new]), at("artery", a2["seg_end"][s_new])
    page.mouse.click((ax + bx) / 2, (ay + by) / 2)
    page.wait_for_selector("#v3-rad-row:not([hidden])")
    page.fill("#v3-rad", "0.77")
    page.press("#v3-rad", "Enter")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('radius 0.77')", timeout=30000)
    assert tree("artery")["seg_radius"][s_new] == pytest.approx(0.77)
    # delete it, then undo
    page.click("#v3-del")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('vessel deleted')", timeout=30000)
    assert len(tree("artery")["seg_start"]) == len(a2["seg_start"]) - 1
    page.click("#v3-undo")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('undone')", timeout=30000)
    assert len(tree("artery")["seg_start"]) == len(a2["seg_start"])
    # make a vein leaf an extra outlet with the I key
    v = tree("vein")
    deg = np.bincount(v["seg_start"] + v["seg_end"], minlength=len(v["node_x"]))
    leaf = int(next(i for i in np.flatnonzero(deg == 1) if i not in v["inlet_nodes"]))
    page.mouse.click(*at("vein", leaf))
    page.wait_for_function("document.querySelector('#v3-inlet').textContent === 'Make outlet'")
    page.keyboard.press("i")
    page.wait_for_function("document.querySelector('#v3-tree-info').textContent"
                           ".includes('outlet set')", timeout=30000)
    assert len(tree("vein")["inlet_nodes"]) == len(v["inlet_nodes"]) + 1
    # the edited trees solve
    page.click("#v3-run")
    page.wait_for_selector("#v3-flow-card:not([hidden])", timeout=120000)
    assert not errors


@pytest.mark.parametrize("dim", [2, 3])
def test_optimise_in_the_organ_then_use_it_in_the_simulation(page, server, dim):
    """Optimise tunes the current organ setup (2-D or 3-D), loads the best values,
    and '📤 Use optimised trees in simulation' opens them as a runnable project."""
    page.goto(server + "/explorer.html")
    _ready(page)
    if dim == 3:
        page.click("#dim-3")
        page.wait_for_timeout(300)
        _ready(page)
    page.select_option("#fill", "cco")
    _ready(page)
    page.click("#opt-card summary")
    page.fill("#maxiter", "3")
    before = page.input_value("#a-gamma")
    page.click("#btn-opt")
    page.wait_for_function("document.querySelector('#opt-info').textContent.includes('Loaded')",
                           timeout=180000)
    info = page.inner_text("#opt-info")
    assert "% better" in info or "kept yours" in info
    assert page.is_visible("#btn-opt-use")
    assert page.input_value("#a-gamma") != before or "kept yours" in info
    with page.context.expect_page() as new:
        page.click("#btn-opt-use")
    sim = new.value
    sim.wait_for_load_state()
    want = "/?project=" if dim == 2 else "/viewer3d.html?project="
    assert want in sim.url
    if dim == 2:                                  # the workspace opened it, trees editable
        wait_status(sim, "opened")
        assert "artery segments" in sim.inner_text("#node-info")
    else:
        sim.wait_for_selector("#proj-card:not([hidden])")


def test_3d_viewer_shear_thinning_run(page, server):
    """Shear-thinning blood in the 3-D viewer: λ appears with the checkbox and the run
    summary shows the viscosity range; the option hides for the direct model."""
    import httpx

    pid = httpx.post(server + "/explore/project", timeout=60, json={
        "dim": 3, "shape": "heart", "fill": "cco", "n_terminals": 12, "generations": 3,
        "nx": 16, "name": "ui-3d-carreau"}).json()["id"]
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{server}/viewer3d.html?project={pid}")
    page.wait_for_selector("#v3-an-card:not([hidden])")
    assert page.is_hidden("#v3-lambda-row")
    page.check("#v3-shear")
    assert page.is_visible("#v3-lambda-row")
    page.click("#v3-run")
    page.wait_for_function("document.querySelector('#v3-summary').textContent"
                           ".includes('viscosity')", timeout=120000)
    page.select_option("#v3-model", "direct")
    assert page.is_hidden("#v3-shear-row")
    assert not errors
