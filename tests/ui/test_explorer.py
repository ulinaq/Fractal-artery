"""The Tree explorer: anatomical roots per organ, manual roots, 3-D fields."""

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
