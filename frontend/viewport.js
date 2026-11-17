// perfusion — shared canvas viewport: fills the space available, stays crisp on
// HiDPI screens, and gives every 2-D view zoom + pan over a physical domain.
//
//   const vp = createViewport(canvas, { domain: () => [x0, x1, y0, y1], onChange: redraw });
//   vp.toCanvas(x, y) / vp.toDomain(px, py)   — domain <-> CSS-pixel canvas coords
//   vp.begin()                                — reset the HiDPI transform + clear
//   vp.W, vp.H, vp.scale                      — logical size, pixels per domain unit
//
// Gestures: wheel / pinch zooms about the cursor; right- or middle-drag, or
// Space + drag, pans (so left-click stays free for editing); with
// `panWithLeft: () => true` a plain left-drag on empty space pans too. Zoom
// buttons (＋ − ⤢ fit) are added to the canvas' parent. 3-D views pass
// `domain: null` and use only the sizing + `onResize`.

const MIN_ZOOM = 0.5, MAX_ZOOM = 40;

export function createViewport(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  const wrap = canvas.parentElement;
  const vp = {
    canvas, ctx, W: canvas.width, H: canvas.height, dpr: 1,
    zoom: 1, cx: null, cy: null,          // view centre in domain units (null = domain centre)
    scale: 1, panning: false,
  };
  const domain = () => (opts.domain ? opts.domain() : null);
  const changed = () => opts.onChange && opts.onChange();

  // ── sizing ──────────────────────────────────────────────────────────────────
  function resize() {
    const r = wrap.getBoundingClientRect();
    const cs = getComputedStyle(wrap);
    const w = Math.max(200, Math.floor(r.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)));
    const h = Math.max(200, Math.floor(r.height - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)));
    vp.dpr = window.devicePixelRatio || 1;
    vp.W = w; vp.H = h;
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    canvas.width = Math.round(w * vp.dpr); canvas.height = Math.round(h * vp.dpr);
    update();
    if (opts.onResize) opts.onResize(vp);
    changed();
  }

  function update() {
    const d = domain();
    if (!d) return;
    const [x0, x1, y0, y1] = d;
    const fit = Math.min(vp.W / (x1 - x0), vp.H / (y1 - y0)) * 0.94;   // small margin
    vp.scale = fit * vp.zoom;
    if (vp.cx == null) { vp.cx = (x0 + x1) / 2; vp.cy = (y0 + y1) / 2; }
    if (zoomLabel) zoomLabel.textContent = Math.round(vp.zoom * 100) + "%";
  }

  vp.begin = () => {
    ctx.setTransform(vp.dpr, 0, 0, vp.dpr, 0, 0);
    ctx.clearRect(0, 0, vp.W, vp.H);
    update();
  };
  vp.toCanvas = (x, y) => [vp.W / 2 + (x - vp.cx) * vp.scale, vp.H / 2 - (y - vp.cy) * vp.scale];
  vp.toDomain = (px, py) => [vp.cx + (px - vp.W / 2) / vp.scale, vp.cy - (py - vp.H / 2) / vp.scale];
  vp.eventXY = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  // ── zoom / pan ──────────────────────────────────────────────────────────────
  vp.zoomAt = (factor, px = vp.W / 2, py = vp.H / 2) => {
    if (!domain()) return;
    update();
    const [x, y] = vp.toDomain(px, py);
    vp.zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, vp.zoom * factor));
    update();
    const [x2, y2] = vp.toDomain(px, py);          // keep the point under the cursor fixed
    vp.cx += x - x2; vp.cy += y - y2;
    changed();
  };
  vp.panBy = (dpx, dpy) => { vp.cx -= dpx / vp.scale; vp.cy += dpy / vp.scale; changed(); };
  vp.fit = () => { vp.zoom = 1; vp.cx = vp.cy = null; update(); changed(); };

  let spaceDown = false, pan = null;
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(e.target.tagName)) {
      if (!spaceDown) canvas.style.cursor = "grab";
      spaceDown = true; e.preventDefault();
    }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") { spaceDown = false; canvas.style.cursor = ""; } });

  if (opts.domain) {
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [px, py] = vp.eventXY(e);
      vp.zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), px, py);   // ctrlKey = pinch
    }, { passive: false });

    // capture phase: a pan gesture is claimed before the app's own handlers see it
    canvas.addEventListener("pointerdown", (e) => {
      const left = e.button === 0;
      const wantPan = e.button === 1 || e.button === 2 || (left && spaceDown)
        || (left && opts.panWithLeft && opts.panWithLeft(e));
      if (!wantPan) return;
      e.stopImmediatePropagation(); e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      pan = { x: e.clientX, y: e.clientY };
      vp.panning = true; canvas.style.cursor = "grabbing";
    }, { capture: true });
    canvas.addEventListener("pointermove", (e) => {
      if (!pan) return;
      e.stopImmediatePropagation();
      vp.panBy(e.clientX - pan.x, e.clientY - pan.y);
      pan = { x: e.clientX, y: e.clientY };
    }, { capture: true });
    const endPan = (e) => {
      if (!pan) return;
      e.stopImmediatePropagation();
      pan = null; vp.panning = false; canvas.style.cursor = spaceDown ? "grab" : "";
    };
    canvas.addEventListener("pointerup", endPan, { capture: true });
    canvas.addEventListener("pointercancel", endPan, { capture: true });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // ── zoom buttons ────────────────────────────────────────────────────────────
  let zoomLabel = null;
  if (opts.domain && opts.buttons !== false) {
    const bar = document.createElement("div");
    bar.className = "zoombar";
    bar.innerHTML = `<button title="zoom in (scroll up)">＋</button><span class="zl">100%</span>`
      + `<button title="zoom out (scroll down)">−</button><button title="fit the whole domain">⤢</button>`;
    const [bIn, lbl, bOut, bFit] = bar.children;
    zoomLabel = lbl;
    bIn.addEventListener("click", () => vp.zoomAt(1.25));
    bOut.addEventListener("click", () => vp.zoomAt(0.8));
    bFit.addEventListener("click", () => vp.fit());
    wrap.appendChild(bar);
  }

  new ResizeObserver(() => resize()).observe(wrap);
  window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener?.("change", resize);
  resize();
  return vp;
}
