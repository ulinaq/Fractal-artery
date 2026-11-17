"""Browser tests for the web app (Playwright).

A real uvicorn server runs the app (with an isolated on-disk store) in a thread,
and a headless browser drives the actual UI — the same checks that were done by
hand while building the editors/explorer. Skipped unless the ``ui`` extra is
installed and a Chromium is available:

    uv run --extra api --extra image --extra ui playwright install chromium
    uv run --extra api --extra image --extra ui pytest tests/ui
"""

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    import httpx
    import uvicorn

    from perfusion.api.app import create_app
    from perfusion.api.store import ProjectStore

    app = create_app(store=ProjectStore(tmp_path_factory.mktemp("ui-store")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        pytest.fail("the API server did not start")
    yield base
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        b = None
        for kw in ({}, {"channel": "chrome"}):          # bundled Chromium, else Chrome
            try:
                b = p.chromium.launch(headless=True, **kw)
                break
            except Exception:  # try the next browser
                continue
        if b is None:
            pytest.skip("no Chromium available (run: playwright install chromium)")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1500, "height": 950})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    yield pg
    ctx.close()


class Canvas:
    """Maps domain coordinates to page pixels through the app's own viewport, so
    clicks land exactly where the app thinks they are at any zoom/pan."""

    def __init__(self, page, module="/core.js"):
        self.page, self.module = page, module

    def at(self, x, y):
        box = self.page.locator("#view").bounding_box()
        px, py = self.page.evaluate(
            f"([x, y]) => import('{self.module}').then(m => m.vp.toCanvas(x, y))", [x, y])
        return box["x"] + px, box["y"] + py

    def click(self, x, y, **kw):
        self.page.mouse.click(*self.at(x, y), **kw)

    def path(self, *pts):
        """Click each point in turn (draw / extend a vessel), then finish (Esc)."""
        for x, y in pts:
            self.click(x, y)
        self.page.keyboard.press("Escape")

    def drag(self, a, b, button="left"):
        self.page.mouse.move(*self.at(*a))
        self.page.mouse.down(button=button)
        self.page.mouse.move(*self.at(*b), steps=8)
        self.page.mouse.up(button=button)


def wait_status(page, *words, timeout=60000):
    js = ("ws => ws.some(w => document.querySelector('#status').textContent.includes(w))")
    page.wait_for_function(js, arg=list(words), timeout=timeout)
