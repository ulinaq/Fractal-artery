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
