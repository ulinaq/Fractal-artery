"""The 2-D workspace: vessel network editing, tissue editing, zoom-accurate
clicks, solving, pruning + undo."""

import pytest

from .conftest import Canvas, wait_status


def _new_project(page, server):
    page.goto(server + "/")
    page.wait_for_selector("#status.ok")
    page.click("#btn-new")
    wait_status(page, "project ready")
    return page.evaluate("new URL(document.querySelector('#proj-id a').href)"
                         ".searchParams.get('project')")


def _editable(page, server, pid):
    return page.evaluate(f"fetch('{server}/projects/{pid}/editable').then(r => r.json())")


def test_draw_branch_resize_inlet_and_save(page, server):
    pid = _new_project(page, server)
    cv = Canvas(page)
    cv.path((30, 10), (30, 50), (30, 90))
    cv.path((30, 50), (10, 80))
    page.keyboard.press("v")
    cv.path((70, 110), (70, 60), (70, 15))
    cv.click(30, 30)                                             # select a segment
    assert "segment" in page.inner_text("#node-info")
    page.fill("#draw-radius", "4.5")
    page.dispatch_event("#draw-radius", "input")
    page.dispatch_event("#draw-radius", "change")
    page.click("#btn-save-vessels")
    wait_status(page, "vessels saved")
    ed = _editable(page, server, pid)
    assert len(ed["artery"]["seg_start"]) == 3
    assert len(ed["vein"]["seg_start"]) == 2
    assert 4.5 in [round(r, 2) for r in ed["artery"]["seg_radius"]]
    assert ed["artery"]["inlet_nodes"] == [0]                    # the first click feeds it
    assert not page.errors


def test_clicks_stay_accurate_when_zoomed_and_panned(page, server):
    pid = _new_project(page, server)
    box = page.locator("#view").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    for _ in range(3):
        page.mouse.wheel(0, -300)
    page.mouse.move(cx, cy)
    page.mouse.down(button="right")
    page.mouse.move(cx - 70, cy + 30, steps=5)
    page.mouse.up(button="right")
    zoom = page.evaluate("import('/core.js').then(m => m.vp.zoom)")
    assert zoom > 1.5
    cv = Canvas(page)
    cv.path((45, 55), (60, 70))
    page.click("#btn-save-vessels")
    wait_status(page, "vessels saved")
    a = _editable(page, server, pid)["artery"]
    pts = sorted(zip(a["node_x"], a["node_y"], strict=True))
    assert pts[0] == pytest.approx((45, 55), abs=0.05)
    assert pts[1] == pytest.approx((60, 70), abs=0.05)


def test_organ_preset_then_run_on_drawn_trees(page, server):
    pid = _new_project(page, server)
    cv = Canvas(page)
    cv.path((50, 15), (50, 50), (35, 80))
    page.keyboard.press("v")
    cv.path((55, 105), (55, 70), (70, 45))
    page.keyboard.press("2")                                     # Tissue tab
    page.select_option("#organ-preset", "kidney")
    page.click("#btn-organ")                                     # fetches the outline
    page.wait_for_function("document.querySelector('#tissue-info').textContent"
                           ".startsWith('1 organ')")
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    assert "solved" in page.inner_text("#status")
    res = page.evaluate(f"fetch('{server}/projects/{pid}/results').then(r => r.json())")
    assert res["arterial_inflow"] > 0
    assert res["arterial_inflow"] == pytest.approx(res["venous_outflow"], rel=1e-6)
    mask = page.evaluate(f"fetch('{server}/projects/{pid}/mask').then(r => r.json())")
    assert len(mask["regions"]) == 1                             # the preset was saved
    assert not page.errors


def test_prune_and_undo(page, server):
    pid = _new_project(page, server)
    cv = Canvas(page)
    cv.path((50, 5), (50, 40), (50, 70))
    cv.path((50, 40), (25, 60))
    page.click("summary:has-text('Grow / prune')")
    page.fill("#prune-steps", "1")
    page.click("#btn-prune")
    wait_status(page, "pruned")
    assert len(_editable(page, server, pid)["artery"]["seg_start"]) == 1   # trunk kept
    page.keyboard.press("ControlOrMeta+z")
    assert "3 artery segments" in page.inner_text("#node-info")


def test_compare_runs_with_a_kept_baseline(page, server):
    """Run, keep the result as a baseline, add a lesion, re-run, compare: the table
    shows the change and a Δ map is drawn with the diverging colour bar."""
    import httpx

    pid = _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.fill("#cmp-name", "healthy")
    page.click("#cmp-keep")
    page.wait_for_function("document.querySelector('#cmp-info').textContent.includes('kept')")
    assert page.inner_text("#cmp-snap") == "healthy"

    dom = page.evaluate("[...document.querySelectorAll('#dom-w, #dom-h')].map(e => +e.value)")
    w, h = dom
    lesion = [[0.3 * w, 0.3 * h], [0.7 * w, 0.3 * h], [0.7 * w, 0.7 * h], [0.3 * w, 0.7 * h]]
    httpx.put(f"{server}/projects/{pid}/lesions",
              json={"lesions": [{"polygon": lesion, "permeability": 0.02}]})
    page.goto(f"{server}/?project={pid}")                        # reopen with the lesion
    wait_status(page, "opened")
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.wait_for_function("!document.querySelector('#cmp-run').disabled")
    page.click("#cmp-run")
    page.wait_for_selector("#cmp-out table.cmp")
    out = page.inner_text("#cmp-out")
    assert "healthy" in out
    assert "arterial inflow" in out
    assert page.is_visible("#cmp-maps")
    page.click('#cmp-maps [data-delta="flux"]')
    page.wait_for_function("document.querySelector('#cb-title').textContent === 'Δ flux'")
    ramp = page.eval_on_selector("#cb-ramp", "e => e.style.background")
    assert "rgb(33, 102, 172)" in ramp                           # the diverging ramp
    assert not page.errors


def test_batch_experiment_sweep_table(page, server):
    """Run a p_in sweep from the Result tab: the sweep table lists each value with its
    inflow, and clicking a row compares against that run."""
    _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.select_option("#cmp-bparam", "p_in")
    page.fill("#cmp-bvals", "5, 20")
    page.click("#cmp-brun")
    page.wait_for_function("document.querySelector('#cmp-info').textContent.includes('runs kept')",
                           timeout=120000)
    rows = page.locator("#cmp-bout tr[data-snap]")
    assert rows.count() == 2
    assert page.inner_text("#cmp-snap").split("\n") == ["p_in = 5", "p_in = 20"]
    rows.nth(1).click()
    page.wait_for_selector("#cmp-out table.cmp")
    assert "p_in = 20" in page.inner_text("#cmp-out")
    assert page.input_value("#cmp-snap") == rows.nth(1).get_attribute("data-snap")
    assert not page.errors


def test_dce_fit_from_the_projects_own_tracer(page, server):
    """Compute the tracer, take its curve as 'measured' data and fit p_in + porosity:
    the table, R² and the measured-vs-model chart appear."""
    _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.click("#btn-flow")
    wait_status(page, "flow ready", timeout=120000)
    page.click("#dce-card summary")
    page.click("#dce-own")
    page.wait_for_function("document.querySelector('#dce-csv').value.split('\\n').length > 10")
    page.click("#dce-go")
    page.wait_for_selector("#dce-out table.cmp", timeout=180000)
    out = page.inner_text("#dce-out")
    assert "p_in" in out
    assert "porosity" in out
    import re

    assert float(re.search(r"R² = ([0-9.]+)", out).group(1)) > 0.98
    assert "±" in out
    assert page.is_visible("#dce-chart")
    assert not page.errors


def test_add_loops_then_solve(page, server):
    """⟳ Add loops joins branches of the chosen tree (terminals kept) and the looped
    model solves."""
    pid = _new_project(page, server)
    page.click("summary:has-text('Grow / prune')")
    page.fill("#loop-n", "5")
    page.click("#btn-loops")
    page.wait_for_function("document.querySelector('#loop-info').textContent.includes('loops')")
    assert "5 loops" in page.inner_text("#loop-info")
    e = _editable(page, server, pid)
    assert e["artery"] is not None
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    assert "solved" in page.inner_text("#status")
    assert not page.errors


def test_shear_thinning_run_reports_the_viscosity_range(page, server):
    """Tick shear-thinning blood: its λ shows, the run reports the Carreau viscosity
    range, and the option hides for the direct-connection model."""
    _new_project(page, server)
    assert page.is_hidden("#lambda-row")
    page.check("#shear-thin")
    assert page.is_visible("#lambda-row")
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    # the summary fills in just after the "solved" status
    page.wait_for_function("document.querySelector('#summary').textContent"
                           ".includes('viscosity × max')", timeout=30000)
    page.select_option("#run-model", "direct")
    assert page.is_hidden("#shear-row")
    assert page.is_hidden("#lambda-row")
    assert not page.errors


def test_casson_run_reports_plugged_vessels(page, server):
    """Pick the Casson law: λ gives way to the yield stress, and a yield stress far
    above the wall stresses plugs vessels, which the summary reports."""
    _new_project(page, server)
    assert page.is_hidden("#law-row")
    page.check("#shear-thin")
    page.select_option("#blood-law", "casson")
    assert page.is_hidden("#lambda-row")
    assert page.is_visible("#yield-row")
    page.fill("#casson-yield", "10")
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.wait_for_function("document.querySelector('#summary').textContent"
                           ".includes('plugged vessels')", timeout=30000)
    page.select_option("#run-model", "direct")
    assert page.is_hidden("#law-row")
    assert page.is_hidden("#yield-row")
    assert not page.errors


def test_two_input_grid_experiment(page, server):
    """A p_in × K grid from the Result tab: a matrix of runs shaded by the chosen
    quantity with its sensitivities; switching the quantity re-renders; clicking a
    cell compares against that run."""
    _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.select_option("#cmp-bparam", "p_in")
    page.fill("#cmp-bvals", "5, 20")
    assert page.is_hidden("#cmp-bvals2")
    page.select_option("#cmp-bparam2", "permeability")
    assert page.is_visible("#cmp-bvals2")
    page.fill("#cmp-bvals2", "0.5, 1, 2")
    page.click("#cmp-brun")
    page.wait_for_function("document.querySelector('#cmp-info').textContent.includes('runs kept')",
                           timeout=120000)
    cells = page.locator("#cmp-bout table.grid td[data-snap]")
    assert cells.count() == 6
    assert "p_in" in page.text_content("#cmp-bout p.sens")
    page.select_option("#cmp-bout select[data-metric-exp]", "mtt")
    page.wait_for_function("document.querySelector('#cmp-bout p.sens').textContent"
                           ".startsWith('MTT')")
    cells.nth(5).click()
    page.wait_for_selector("#cmp-out table.cmp")
    assert "p_in = 20, permeability = 2" in page.inner_text("#cmp-out")
    assert not page.errors


def test_grow_with_loop_closure(page, server):
    """Space colonisation with "close loops while growing": the grow info counts the
    loops and the looped tree solves; the option hides for CCO."""
    _new_project(page, server)
    page.click("summary:has-text('Grow / prune')")
    page.select_option("#grow-method", "cco")
    assert page.is_hidden("#grow-loops-row")
    page.select_option("#grow-method", "colonization")
    page.check("#grow-loops")
    page.click("#btn-grow")
    page.wait_for_function("document.querySelector('#grow-info').textContent.includes('loops')",
                           timeout=60000)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    assert "solved" in page.inner_text("#status")
    assert not page.errors


def test_baseline_names_are_shown_as_text_never_markup(page, server):
    """A kept baseline's name is user text: it used to go into innerHTML raw, so a
    name with markup ran script for anyone opening the shared project link."""
    _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    evil = '<img src=x onerror="window.__xss = 1">'
    page.fill("#cmp-name", evil)
    page.click("#cmp-keep")
    page.wait_for_function("document.querySelector('#cmp-info').textContent.includes('kept')")
    assert page.inner_text("#cmp-snap") == evil
    page.click("#cmp-run")
    page.wait_for_selector("#cmp-out table.cmp")
    assert evil in page.inner_text("#cmp-out")
    assert page.evaluate("document.querySelectorAll('#cmp-snap img, #cmp-out img').length") == 0
    assert page.evaluate("window.__xss") is None
    assert not page.errors


def test_rerun_after_viewing_a_derived_map(page, server):
    """An O₂ (or Δ / kinetic) map used to overwrite state.field with its label, so the
    next Run asked the server for the field "O₂ saturation" and failed with a 404."""
    _new_project(page, server)
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    page.click("details:has(#btn-o2-sat) > summary")     # open the (collapsed) O₂ card
    page.click("#btn-o2-sat")
    page.wait_for_function("document.querySelector('#o2-info').textContent.includes('OEF')",
                           timeout=60000)
    assert page.inner_text("#cb-title") == "O₂ saturation"
    page.click("#btn-run")
    wait_status(page, "solved", "failed", timeout=120000)
    # ("solved" shows before the field reloads: wait for the colour bar to follow)
    page.wait_for_function("['pressure', 'flux', 'tof'].includes("
                           "document.querySelector('#cb-title').textContent)", timeout=60000)
    assert "solved" in page.inner_text("#status")
    assert not page.errors


def test_an_interrupted_job_ends_the_run_instead_of_locking_the_ui(page, server):
    """The pollers only stopped on done/error: a job reaped to `interrupted` (server
    restart) spun forever with every button disabled."""
    _new_project(page, server)
    page.route("**/jobs/*", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body='{"id": "x", "state": "interrupted", "project_id": "p", "error": null}'))
    page.click("#btn-run")
    wait_status(page, "failed", timeout=30000)
    assert "restarted" in page.inner_text("#run-info")
    assert not page.is_disabled("#btn-run")             # the UI is usable again
    assert not page.errors
