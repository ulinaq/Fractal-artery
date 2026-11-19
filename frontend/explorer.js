// perfusion — tree explorer. Grow an arterial + venous tree into a tissue shape
// (2-D or 3-D) from user-placed roots: a fractal skeleton of N generations, then an
// optional space-filling growth (colonisation steps / CCO terminals). Every change
// re-grows (and optionally re-solves) via POST /explore/grow. Self-contained.
import { VIRIDIS, viridis, viridisCss } from "./palette.js";
import { createViewport } from "./viewport.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
// 2-D: the physical domain. 3-D: points are projected to normalised screen
// coordinates (±0.5 spans the box) first, so the same viewport zooms/pans both.
const V3 = [-0.62, 0.62, -0.62, 0.62];

const status = (t, k = "") => { $("status").textContent = t; $("status").className = "status " + k; };
const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
const num = (id) => Number($(id).value);
const optNum = (id) => ($(id).value === "" ? null : Number($(id).value));

// ── state ─────────────────────────────────────────────────────────────────────
const DEFAULTS = {
  2: { art: { root: [0, 8], angle: 90 }, ven: { root: [0, 152], angle: -90 } },
  3: { art: { root: [30, 30, 3], dir: "0,0,1" }, ven: { root: [30, 30, 57], dir: "0,0,-1" } },
};
const DIRS = [["0,0,1", "up (+z)"], ["0,0,-1", "down (−z)"], ["1,0,0", "+x"], ["-1,0,0", "−x"],
              ["0,1,0", "+y"], ["0,-1,0", "−y"], ["center", "toward the centre"]];
const S = {
  dim: 2,
  vp: null,
  polygon: null,          // custom 2-D outline (null = preset)
  data: null,             // last /explore/grow response
  sel: null,              // selected outline vertex index
  autoRoots: true,        // roots follow the organ's anatomy until the user moves one
  lesions: [],            // {center, radius, permeability, density} — discs / spheres
  view: { yaw: 0.7, pitch: 0.35, zoom: 1 },
};

// ── request ───────────────────────────────────────────────────────────────────
function dirOf(prefix, root) {
  const v = $(prefix + "-dir").value;
  if (v !== "center") return v.split(",").map(Number);
  const d = [30 - root[0], 30 - root[1], 30 - root[2]];
  return Math.hypot(...d) > 1e-6 ? d : [0, 0, 1];
}

function treeReq(prefix) {
  const t = { radius: num(prefix + "-radius"), gamma: num(prefix + "-gamma") };
  if (S.autoRoots) return t;                   // the server places the organ's roots
  t.root = S.dim === 3 ? [num(prefix + "-x"), num(prefix + "-y"), num(prefix + "-z")]
                       : [num(prefix + "-x"), num(prefix + "-y")];
  const root = t.root;
  if (S.dim === 3) t.direction = dirOf(prefix, root);
  else t.angle = num(prefix + "-angle");
  return t;
}

function request(over = {}) {
  const d3 = S.dim === 3;
  const shape = d3 ? $("shape3").value : (S.polygon ? "custom" : $("shape2").value);
  return {
    dim: S.dim, shape, polygon: !d3 && S.polygon ? S.polygon : null, size: num("size"),
    extent: num("extent"), twist: num("twist"), jitter: num("jitter"),
    artery: treeReq("a"), vein: treeReq("v"),
    generations: num("generations"), length_to_radius: num("ltr"), exponent: num("exponent"),
    fill: $("fill").value, fill_target: $("fill-target").value,
    fill_steps: num("fill-steps"), n_points: num("n-points"),
    step: optNum("step-len"), influence: optNum("influence"),
    close_loops: $("close-loops").checked, lesions: S.lesions,
    n_terminals: num("n-terminals"), seed: num("seed"),
    solve: $("solve").checked, field: $("field").value, kind: $("kind").value,
    ...gridSize(), ...over,
  };
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" },
                                body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = (await r.text()).slice(0, 200);
    try {
      const j = JSON.parse(msg);
      msg = typeof j.detail === "string" ? j.detail
        : Array.isArray(j.detail)                 // pydantic: "maxiter: Input should be …"
          ? j.detail.map((d) => `${(d.loc || []).filter((l) => l !== "body").join(".")}: ${d.msg}`).join("; ")
          : JSON.stringify(j.detail);
    } catch { /* raw */ }
    throw new Error(msg);
  }
  return r.json();
}

// coalesce bursts of changes into one in-flight request + one queued
let busy = false, queued = null;
async function evaluate(over = {}) {
  if (busy) { queued = over; return; }
  busy = true;
  syncLabels();
  status(over.solve === false || !$("solve").checked ? "growing…" : "growing + solving…");
  try {
    setData(await post("/explore/grow", request(over)));
    showStats();
    status(S.data.solve_error ? S.data.solve_error : "ready", S.data.solve_error ? "err" : "ok");
  } catch (e) {
    status(e.message, "err");
  } finally {
    busy = false;
    if (queued) { const q = queued; queued = null; evaluate(q); }
  }
}

let debounce = null;
const schedule = (ms = 250) => { clearTimeout(debounce); debounce = setTimeout(() => evaluate(), ms); };

// New data; when its physical window changed (first load, resize, 2-D/3-D) the
// view refits so the whole domain is in view — zoom/pan are kept otherwise.
function setData(d) {
  S.data = d;
  syncRootInputs(d);
  const key = `${d.dim}:${d.domain.join(",")}`;
  if (key !== S.fitKey) { S.fitKey = key; S.vp.fit(); }   // fit() redraws
  else draw();
}

// show the roots actually used (anatomical defaults included) in the inputs
function syncRootInputs(d) {
  for (const [p, kind] of [["a", "artery"], ["v", "vein"]]) {
    const r = d.roots[kind], aim = d.aims && d.aims[kind];
    $(p + "-x").value = +r[0].toFixed(1); $(p + "-y").value = +r[1].toFixed(1);
    if (d.dim === 3) {
      $(p + "-z").value = +r[2].toFixed(1);
      if (aim && S.autoRoots) setAnatomicalDir(p, aim);
    } else if (aim != null) {
      $(p + "-angle").value = Math.round(aim);
    }
  }
  $("root-mode").textContent = S.autoRoots ? "roots: organ default (anatomical)" : "roots: placed by you";
}

// 3-D: the organ's own trunk direction appears as an "anatomical" choice
function setAnatomicalDir(p, v) {
  const sel = $(p + "-dir"), val = v.map((c) => c.toFixed(3)).join(",");
  let opt = sel.querySelector("option[data-anat]");
  if (!opt) { opt = document.createElement("option"); opt.dataset.anat = "1"; sel.prepend(opt); }
  opt.value = val;
  opt.textContent = `anatomical (${v.map((c) => c.toFixed(2)).join(", ")})`;
  sel.value = val;
}

const manualRoots = () => { S.autoRoots = false; };
function organDefaultRoots() { S.autoRoots = true; evaluate(); }

// ── drawing ───────────────────────────────────────────────────────────────────
function draw() {
  const vp = S.vp;
  if (!vp) return;                 // the viewport's first resize fires before it's assigned
  vp.begin();
  ctx.fillStyle = S.dim === 3 ? "#0b1020" : "#f1f5f9";
  ctx.fillRect(0, 0, vp.W, vp.H);
  if (!S.data || S.data.dim !== S.dim) return;
  if (S.data.dim === 3) draw3d(); else draw2d();
}

// 2-D: domain ↔ canvas (zoom + pan included)
const tc = (x, y) => S.vp.toCanvas(x, y);
const td = (px, py) => S.vp.toDomain(px, py);
// line width for a vessel radius, thickening as you zoom in
const rw = (r, k, min) => Math.max(min, r * k * S.vp.scale / (S.dim === 3 ? 520 : 3.5));

const HANDLE_LEN = 14;   // domain units from root to its direction handle

function dirHandle(prefix) {
  const r = [num(prefix + "-x"), num(prefix + "-y")], a = num(prefix + "-angle") * Math.PI / 180;
  return [r[0] + HANDLE_LEN * Math.cos(a), r[1] + HANDLE_LEN * Math.sin(a)];
}

function draw2d() {
  const d = S.data;
  const [dx0, dx1, dy0, dy1] = d.domain;                     // the domain sheet
  const [sx0, sy0] = tc(dx0, dy1), [sx1, sy1] = tc(dx1, dy0);
  ctx.fillStyle = "#fff"; ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
  ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1; ctx.strokeRect(sx0 + 0.5, sy0 + 0.5, sx1 - sx0, sy1 - sy0);
  if (d.field) paintHeatmap(d.field.data, d.field.shape[0], d.field.shape[1], d.field.vmin, d.field.vmax);
  else $("colorbar").hidden = true;
  // tissue outline
  ctx.beginPath();
  d.outline.forEach(([x, y], i) => { const [px, py] = tc(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.closePath();
  if (!d.field) { ctx.fillStyle = "rgba(16,185,129,.10)"; ctx.fill(); }
  ctx.strokeStyle = "rgba(5,150,105,.85)"; ctx.lineWidth = 1.5; ctx.stroke();
  // trees
  ctx.lineCap = "round";
  for (const [ax, ay, bx, by, r, kind] of d.segments) {
    const [p0x, p0y] = tc(ax, ay), [p1x, p1y] = tc(bx, by);
    ctx.strokeStyle = kind === 1 ? "rgba(37,99,235,.9)" : "rgba(220,38,38,.9)";
    ctx.lineWidth = rw(r, 0.9, 0.6);
    ctx.beginPath(); ctx.moveTo(p0x, p0y); ctx.lineTo(p1x, p1y); ctx.stroke();
  }
  // outline vertex handles
  d.outline.forEach(([x, y], i) => {
    const [px, py] = tc(x, y), s = S.sel === i ? 5 : 3.5;
    ctx.fillStyle = S.sel === i ? "#2563eb" : "#fff";
    ctx.strokeStyle = S.sel === i ? "#2563eb" : "#047857"; ctx.lineWidth = 1.4;
    ctx.fillRect(px - s, py - s, 2 * s, 2 * s); ctx.strokeRect(px - s, py - s, 2 * s, 2 * s);
  });
  // lesions: dashed discs with a centre handle and a rim (radius) handle
  S.lesions.forEach((le, i) => {
    const [cx, cy] = tc(le.center[0], le.center[1]);
    const [rx] = tc(le.center[0] + le.radius, le.center[1]);
    lesionCircle(cx, cy, rx - cx, i);
    ctx.beginPath(); ctx.arc(rx, cy, 4.5, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke();
  });
  // roots + direction handles
  rootMarker("a", true, "#dc2626"); rootMarker("v", false, "#2563eb");
}

const LESION_COLOR = "#a21caf";
function lesionCircle(cx, cy, r, i) {
  const col = S.dim === 3 ? "#f0abfc" : LESION_COLOR;   // light on the dark 3-D view
  ctx.strokeStyle = col; ctx.lineWidth = S.dim === 3 ? 2.5 : 1.8; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.arc(cx, cy, Math.max(r, 2), 0, 7); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = S.dim === 3 ? "rgba(240,171,252,.12)" : "rgba(162,28,175,.08)"; ctx.fill();
  ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 7); ctx.fillStyle = col; ctx.fill();
  ctx.fillStyle = col; ctx.font = "600 11px system-ui, sans-serif";
  ctx.fillText(`L${i + 1}`, cx + 7, cy - 7);
}

function rootMarker(prefix, up, color) {
  const [rx, ry] = tc(num(prefix + "-x"), num(prefix + "-y"));
  const [hx, hy] = tc(...dirHandle(prefix));
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(hx, hy); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.arc(hx, hy, 5, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke();
  const s = 8;
  ctx.beginPath();
  if (up) { ctx.moveTo(rx, ry - s); ctx.lineTo(rx - s, ry + s); ctx.lineTo(rx + s, ry + s); }
  else { ctx.moveTo(rx, ry + s); ctx.lineTo(rx - s, ry - s); ctx.lineTo(rx + s, ry - s); }
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
}

function paintHeatmap(data, nx, ny, vmin, vmax) {
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(nx, ny);
  const span = vmax - vmin || 1;
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const v = data[ix][iy];
      const o = ((ny - 1 - iy) * nx + ix) * 4;
      if (v === null) { img.data[o + 3] = 0; continue; }
      const [r, g, b] = viridis((v - vmin) / span);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const [x0, x1, y0, y1] = S.data.domain;
  const [px0, py0] = tc(x0, y1), [px1, py1] = tc(x1, y0);
  ctx.imageSmoothingEnabled = S.vp.zoom < 3;               // zoomed in: show the grid cells
  ctx.drawImage(off, px0, py0, px1 - px0, py1 - py0);
  updateColorbar(vmin, vmax, S.data.field.name);
}

// 3-D: orthographic projection, yaw about z then pitch about x; z is up on screen.
// Normalised so the box spans about ±0.5, then mapped through the viewport.
function project(x, y, z) {
  const [x0, x1, y0, y1, z0, z1] = S.data.domain;
  const span = Math.max(x1 - x0, y1 - y0, z1 - z0);
  let dx = (x - (x0 + x1) / 2) / span, dy = (y - (y0 + y1) / 2) / span, dz = (z - (z0 + z1) / 2) / span;
  const cy = Math.cos(S.view.yaw), sy = Math.sin(S.view.yaw);
  [dx, dy] = [dx * cy - dy * sy, dx * sy + dy * cy];
  const cp = Math.cos(S.view.pitch), sp = Math.sin(S.view.pitch);
  [dy, dz] = [dy * cp - dz * sp, dy * sp + dz * cp];
  const [sx, sy2] = S.vp.toCanvas(dx, dz);
  return [sx, sy2, dy];
}

function draw3d() {
  const d = S.data;
  // domain box wireframe
  const [x0, x1, y0, y1, z0, z1] = d.domain;
  const C = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]].map((p) => project(...p));
  ctx.strokeStyle = "rgba(148,163,184,.25)"; ctx.lineWidth = 1;
  for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) {
    ctx.beginPath(); ctx.moveTo(C[a][0], C[a][1]); ctx.lineTo(C[b][0], C[b][1]); ctx.stroke();
  }
  // tissue: solved pressure voxels, else a faint point cloud of the shape
  if (d.field) {
    const { min, max, voxels } = d.field, span = (max - min) || 1;
    const pts = voxels.map((v) => { const [sx, sy, dep] = project(v[0], v[1], v[2]); return { sx, sy, dep, t: (v[3] - min) / span }; })
      .sort((a, b) => a.dep - b.dep);
    ctx.globalAlpha = 0.35;
    const r = Math.max(1.2, S.vp.scale * 0.0075);          // voxel dot grows with zoom
    for (const p of pts) {
      ctx.fillStyle = viridisCss(Math.max(0, Math.min(1, p.t)));
      ctx.fillRect(p.sx - r, p.sy - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
    updateColorbar(min, max, d.field.name);
  } else {
    $("colorbar").hidden = true;
    ctx.fillStyle = "rgba(52,211,153,.35)";
    for (const v of d.tissue_points || []) { const [sx, sy] = project(...v); ctx.fillRect(sx - 1, sy - 1, 2, 2); }
  }
  // trees, far segments first
  const segs = d.segments.map((s) => {
    const a = project(s[0], s[1], s[2]), b = project(s[3], s[4], s[5]);
    return { a, b, r: s[6], k: s[7], dep: (a[2] + b[2]) / 2 };
  }).sort((p, q) => p.dep - q.dep);
  ctx.lineCap = "round";
  for (const s of segs) {
    ctx.strokeStyle = s.k === 0 ? "#ff6b6b" : "#5b8dff";
    ctx.lineWidth = rw(s.r, 0.9, 0.8);
    ctx.beginPath(); ctx.moveTo(s.a[0], s.a[1]); ctx.lineTo(s.b[0], s.b[1]); ctx.stroke();
  }
  S.lesions.forEach((le, i) => {                 // spheres, drawn as their outline
    const [sx, sy] = project(...le.center);
    const span = Math.max(x1 - x0, y1 - y0, z1 - z0);
    const r = Math.abs(S.vp.toCanvas(le.radius / span, 0)[0] - S.vp.toCanvas(0, 0)[0]);
    lesionCircle(sx, sy, r, i);
  });
  for (const [kind, color] of [["artery", "#ff6b6b"], ["vein", "#5b8dff"]]) {
    const [sx, sy] = project(...d.roots[kind]);
    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, 7); ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function updateColorbar(vmin, vmax, label) {
  $("colorbar").hidden = false;
  const stops = VIRIDIS.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i / (VIRIDIS.length - 1)).toFixed(0)}%`);
  $("cb-ramp").style.background = `linear-gradient(to top, ${stops.join(",")})`;
  $("cb-max").textContent = fmt(vmax); $("cb-min").textContent = fmt(vmin);
  $("cb-title").textContent = label;
}

function showStats() {
  const d = S.data, a = d.stats.artery, v = d.stats.vein;
  const stat = (val, lbl) => `<div class="stat"><b>${val}</b><span>${lbl}</span></div>`;
  let h = stat(`${a.segments} / ${v.segments}`, "segments art / ven")
    + stat(`${a.terminals} / ${v.terminals}`, "terminals art / ven")
    + stat(`${a.max_order} / ${v.max_order}`, "Strahler order art / ven")
    + (a.loops || v.loops ? stat(`${a.loops} / ${v.loops}`, "loops art / ven") : "")
    + stat(d.active_cells, "tissue cells");
  if (d.inflow != null) h += stat(fmt(d.inflow), "perfused inflow");
  if (d.objective != null) h += stat(fmt(d.objective), d.kind + (S.lesions.length ? " (healthy)" : ""));
  if (d.lesion) {
    const L = d.lesion;
    if (L.flux_ratio != null) h += stat(L.flux_ratio.toFixed(2) + "×", "lesion flux / healthy");
    h += stat(L.terminals_inside, "terminals in lesions");
  }
  $("summary").innerHTML = h;
  $("lesion-info").textContent = !S.lesions.length ? ""
    : d.lesion ? `${d.lesion.cells} lesion cells · flux ${d.lesion.flux_ratio != null ? d.lesion.flux_ratio.toFixed(2) + "× healthy" : "–"}`
      + ` · ${d.lesion.terminals_inside} vessel terminals inside`
    : "solve to see the lesion's perfusion";
  $("obj-info").textContent = d.objective != null ? `${d.kind} = ${fmt(d.objective)}`
    : ($("solve").checked ? "—" : "solving is off");
}

// ── 2-D canvas interaction: roots, aim handles, outline points ─────────────────
const canvasXY = (e) => S.vp.eventXY(e);
let drag = null;     // {what: "a-root"|"v-root"|"a-aim"|"v-aim"|"vertex"|"rotate", i, lx, ly}

function hit2d(px, py) {
  const near = (x, y, tol) => { const [cx, cy] = tc(x, y); return (cx - px) ** 2 + (cy - py) ** 2 < tol * tol; };
  for (const p of ["a", "v"]) {
    if (near(...dirHandle(p), 9)) return { what: p + "-aim" };
    if (near(num(p + "-x"), num(p + "-y"), 11)) return { what: p + "-root" };
  }
  for (let i = S.lesions.length - 1; i >= 0; i--) {
    const le = S.lesions[i];
    if (near(le.center[0] + le.radius, le.center[1], 8)) return { what: "lesion-rim", i };
    if (near(le.center[0], le.center[1], 9)) return { what: "lesion", i };
  }
  const o = S.data.outline;
  for (let i = 0; i < o.length; i++) if (near(o[i][0], o[i][1], 8)) return { what: "vertex", i };
  return null;
}

function ensureCustom() {
  manualRoots();                 // keep the roots where they are while reshaping
  if (!S.polygon) { S.polygon = S.data.outline.map((p) => [...p]); $("shape2").value = "custom"; }
}

canvas.addEventListener("pointerdown", (e) => {
  if (!S.data) return;
  canvas.setPointerCapture(e.pointerId);
  const [px, py] = canvasXY(e);
  if (S.data.dim === 3) { drag = { what: "rotate", lx: px, ly: py }; return; }
  const h = hit2d(px, py);
  S.sel = h && h.what === "vertex" ? h.i : null;
  drag = h;
  draw();
});

canvas.addEventListener("pointermove", (e) => {
  const [px, py] = canvasXY(e);
  if (!drag) {
    if (S.data && S.data.dim === 2) canvas.style.cursor = hit2d(px, py) ? "grab" : "default";
    return;
  }
  if (drag.what === "rotate") {
    S.view.yaw += (px - drag.lx) * 0.01; S.view.pitch += (py - drag.ly) * 0.01;
    S.view.pitch = Math.max(-1.5, Math.min(1.5, S.view.pitch));
    drag.lx = px; drag.ly = py; draw(); return;
  }
  const [x, y] = td(px, py);
  if (drag.what.endsWith("-root") || drag.what.endsWith("-aim")) manualRoots();
  if (drag.what.endsWith("-root")) {
    const p = drag.what[0];
    $(p + "-x").value = x.toFixed(1); $(p + "-y").value = y.toFixed(1);
  } else if (drag.what.endsWith("-aim")) {
    const p = drag.what[0];
    const ang = Math.atan2(y - num(p + "-y"), x - num(p + "-x")) * 180 / Math.PI;
    $(p + "-angle").value = Math.round(ang);
  } else if (drag.what === "lesion") {
    S.lesions[drag.i].center = [+x.toFixed(1), +y.toFixed(1)];
    renderLesions(); draw(); return;              // regrow on release
  } else if (drag.what === "lesion-rim") {
    const c = S.lesions[drag.i].center;
    S.lesions[drag.i].radius = +Math.max(1, Math.hypot(x - c[0], y - c[1])).toFixed(1);
    renderLesions(); draw(); return;
  } else if (drag.what === "vertex") {
    ensureCustom();
    S.polygon[drag.i] = [x, y];
    S.data.outline = S.polygon;
  }
  draw();
  livePreview();
});

const endDrag = () => {
  if (!drag) return;
  const was = drag.what; drag = null;
  if (was !== "rotate") evaluate();        // full grow (+ solve) on release
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

// while dragging, a fast geometry-only regrow keeps the trees following the handle
let lastPreview = 0;
function livePreview() {
  const now = performance.now();
  if (now - lastPreview < 120 || busy) return;
  lastPreview = now;
  evaluate({ solve: false });
}

canvas.addEventListener("dblclick", (e) => {
  if (!S.data || S.data.dim !== 2) return;
  const [px, py] = canvasXY(e);
  const o = S.data.outline;
  let best = -1, bd = 64;
  for (let i = 0; i < o.length; i++) {
    const [ax, ay] = tc(...o[i]), [bx, by] = tc(...o[(i + 1) % o.length]);
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L));
    const dd = (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2;
    if (dd < bd) { bd = dd; best = i; }
  }
  if (best < 0) return;
  ensureCustom();
  S.polygon.splice(best + 1, 0, td(px, py));
  S.sel = best + 1;
  evaluate();
});



document.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  if ((e.key === "Delete" || e.key === "Backspace") && S.sel != null && S.data.dim === 2) {
    ensureCustom();
    if (S.polygon.length <= 3) { status("the tissue outline needs at least 3 points", "err"); return; }
    S.polygon.splice(S.sel, 1); S.sel = null;
    e.preventDefault(); evaluate();
  }
});

// ── controls ──────────────────────────────────────────────────────────────────
const LABELS = { size: [2], generations: [0], ltr: [1], exponent: [2], twist: [0], jitter: [2],
                 "fill-steps": [0], "n-terminals": [0], res: [0] };

// grid resolution: 2-D nx × ny (ny keeps the 140:160 aspect), 3-D n³
function gridSize() {
  const n = num("res");
  return S.dim === 3 ? { nx: n, ny: n } : { nx: n, ny: Math.round(n * 8 / 7) };
}
function syncDomainInfo() {
  const e = num("extent"), g = gridSize();
  $("v-extent").textContent = "×" + e.toFixed(2);
  const size = S.dim === 3 ? `${(60 * e).toFixed(0)}³` : `${(140 * e).toFixed(0)} × ${(160 * e).toFixed(0)}`;
  const cells = S.dim === 3 ? `${g.nx}³ = ${(g.nx ** 3).toLocaleString()}` : `${g.nx} × ${g.ny} = ${(g.nx * g.ny).toLocaleString()}`;
  $("res-info").textContent = `domain ${size} units · grid ${cells} cells`;
}
function syncLabels() {
  for (const [id, [dp]] of Object.entries(LABELS)) {
    const el = $("v-" + id);
    if (el) el.textContent = num(id).toFixed(dp);
  }
  syncDomainInfo();
}

function syncVisibility() {
  const d3 = S.dim === 3;
  document.querySelectorAll(".d2").forEach((el) => { el.style.display = d3 ? "none" : ""; });
  document.querySelectorAll(".d3").forEach((el) => { el.style.display = d3 ? "" : "none"; });
  $("dim-2").classList.toggle("active", !d3); $("dim-3").classList.toggle("active", d3);
  const fill = $("fill").value;
  document.querySelector(".fill-col").style.display = fill === "colonization" ? "" : "none";
  document.querySelector(".fill-cco").style.display = fill === "cco" ? "" : "none";
  $("fill-target").disabled = fill === "none";
}

function setRoots(dim) {
  const D = DEFAULTS[dim], e = num("extent");
  for (const [p, t] of [["a", D.art], ["v", D.ven]]) {
    $(p + "-x").value = t.root[0] * e; $(p + "-y").value = t.root[1] * e;
    if (dim === 3) { $(p + "-z").value = t.root[2] * e; $(p + "-dir").value = t.dir; }
    else $(p + "-angle").value = t.angle;
  }
}

// a bigger/smaller object: scale the roots and any custom outline with it
let lastExtent = 4;          // matches the slider's default (the explorer opens at ×4)
function onExtent() {
  const k = num("extent") / lastExtent;
  lastExtent = num("extent");
  // roots, trunk calibre and any explicit colonisation lengths scale with the object
  const ids = ["a-radius", "v-radius", "step-len", "influence"];
  if (!S.autoRoots) ids.push("a-x", "a-y", "a-z", "v-x", "v-y", "v-z");
  for (const id of ids) {
    if ($(id).value !== "") $(id).value = +(num(id) * k).toFixed(2);
  }
  if (S.polygon) S.polygon = S.polygon.map(([x, y]) => [x * k, y * k]);
  for (const le of S.lesions) {                 // lesions scale with the object too
    le.center = le.center.map((c) => +(c * k).toFixed(2)); le.radius = +(le.radius * k).toFixed(2);
  }
  renderLesions();
  syncLabels(); S.vp.fit(); schedule(300);
}

function setDim(dim) {
  S.dim = dim; S.sel = null; S.data = null; S.autoRoots = true;
  S.lesions = []; renderLesions();               // a disc is not a sphere
  S.view.yaw = 0.7; S.view.pitch = 0.35;
  S.vp.fit();
  document.querySelector(".viewhint").textContent = dim === 3
    ? "drag = rotate · scroll = zoom · right-drag / Space+drag = pan · ⤢ = fit"
    : "scroll = zoom · drag empty space / right-drag = pan · ⤢ = fit";
  setRoots(dim);
  // scale the skeleton to the domain: the 3-D box (60³) is smaller than the 2-D
  // window (140×160), and 3-D colonisation grows many segments per step
  const D = dim === 3 ? { ltr: 5, gen: 4, ar: 1.5, vr: 1.8, steps: 25, res: [12, 48, 22] }
                      : { ltr: 10, gen: 4, ar: 2, vr: 2.4, steps: 40, res: [30, 200, 80] };
  const e = num("extent");                     // trunk calibre scales with the object
  $("ltr").value = D.ltr; $("generations").value = D.gen; $("fill-steps").value = D.steps;
  $("res").min = D.res[0]; $("res").max = D.res[1]; $("res").step = dim === 3 ? 1 : 2;
  $("res").value = D.res[2];
  $("a-radius").value = +(D.ar * e).toFixed(2); $("v-radius").value = +(D.vr * e).toFixed(2);
  syncVisibility(); syncLabels();
  evaluate();
}

function stepper(dec, inc, id, by) {
  const bump = (d) => {
    const el = $(id);
    el.value = Math.max(Number(el.min), Math.min(Number(el.max), num(id) + d));
    syncLabels(); evaluate();
  };
  $(dec).addEventListener("click", () => bump(-by));
  $(inc).addEventListener("click", () => bump(by));
}

for (const sel of document.querySelectorAll(".dirsel")) {
  sel.innerHTML = DIRS.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
}

$("dim-2").addEventListener("click", () => setDim(2));
$("dim-3").addEventListener("click", () => setDim(3));
$("shape2").addEventListener("change", () => {
  if ($("shape2").value !== "custom") { S.polygon = null; S.autoRoots = true; }   // a preset organ
  else if (!S.polygon && S.data) S.polygon = S.data.outline.map((p) => [...p]);
  S.sel = null; evaluate();
});
$("size").addEventListener("input", () => {
  if (S.polygon) { S.polygon = null; $("shape2").value = "ellipse"; }   // size rescales presets
  syncLabels(); schedule();
});
// sliders re-grow while dragging (debounced); selects / numbers on change
$("extent").addEventListener("input", onExtent);
const bumpExtent = (f) => {
  const el = $("extent");
  el.value = Math.max(Number(el.min), Math.min(Number(el.max), num("extent") * f));
  onExtent();
};
$("ext-dec").addEventListener("click", () => bumpExtent(0.5));
$("ext-inc").addEventListener("click", () => bumpExtent(2));
for (const id of ["generations", "ltr", "exponent", "twist", "jitter", "fill-steps", "n-terminals", "res"]) {
  $(id).addEventListener("input", () => { syncLabels(); schedule(); });
}
$("shape3").addEventListener("change", () => { S.autoRoots = true; evaluate(); });
for (const id of ["a-dir", "v-dir"]) $(id).addEventListener("change", () => { manualRoots(); evaluate(); });
for (const id of ["fill", "fill-target", "field", "kind"]) {
  $(id).addEventListener("change", () => { syncVisibility(); evaluate(); });
}
for (const id of ["a-x", "a-y", "a-z", "a-angle", "v-x", "v-y", "v-z", "v-angle"]) {
  $(id).addEventListener("change", () => { manualRoots(); evaluate(); });
}
for (const id of ["a-radius", "a-gamma", "v-radius", "v-gamma", "n-points", "step-len",
                  "influence", "close-loops", "seed"]) {
  $(id).addEventListener("change", () => evaluate());
}
$("btn-root-default").addEventListener("click", organDefaultRoots);

// ── lesions: a list of discs / spheres with their own K and vessel density ─────
function tissueCentre() {
  const d = S.data;
  if (S.dim === 2 && d && d.outline) {
    const n = d.outline.length;
    return [d.outline.reduce((a, p) => a + p[0], 0) / n, d.outline.reduce((a, p) => a + p[1], 0) / n];
  }
  const dom = d ? d.domain : [0, 60 * num("extent"), 0, 60 * num("extent"), 0, 60 * num("extent")];
  return S.dim === 3 ? [0, 2, 4].map((k) => (dom[k] + dom[k + 1]) / 2)
                     : [(dom[0] + dom[1]) / 2, (dom[2] + dom[3]) / 2];
}
function addLesion() {
  const c = tissueCentre(), d = S.data;
  const span = d ? Math.min(d.domain[1] - d.domain[0], d.domain[3] - d.domain[2]) : 60 * num("extent");
  S.lesions.push({ center: c.map((v) => +v.toFixed(1)), radius: +(0.15 * span).toFixed(1),
                   permeability: 0.1, density: 0 });
  $("lesion-card").open = true;
  renderLesions(); evaluate();
}
function renderLesions() {
  const axes = S.dim === 3 ? ["x", "y", "z"] : ["x", "y"];
  $("lesion-list").innerHTML = S.lesions.map((le, i) => `
    <div class="lesion" data-i="${i}">
      <div class="roothead"><span class="dot lesion-dot"></span> L${i + 1}
        <button class="lesion-del" data-i="${i}" title="remove this lesion">✕</button></div>
      <div class="grid2">
        ${axes.map((a, k) => `<label>${a} <input type="number" step="1" data-i="${i}" data-k="c${k}" value="${le.center[k]}" /></label>`).join("")}
        <label>radius <input type="number" step="1" min="0.5" data-i="${i}" data-k="radius" value="${le.radius}" /></label>
        <label title="bed permeability relative to healthy tissue (1)">K × <input type="number" step="0.05" min="0.001" data-i="${i}" data-k="permeability" value="${le.permeability}" /></label>
        <label title="vessel density for growth relative to healthy tissue: 0 = avascular (necrotic core), > 1 = angiogenic (tumour rim)">vessel density × <input type="number" step="0.5" min="0" max="20" data-i="${i}" data-k="density" value="${le.density}" /></label>
      </div>
    </div>`).join("");
}
$("btn-lesion-add").addEventListener("click", addLesion);
$("lesion-list").addEventListener("change", (e) => {
  const t = e.target, i = +t.dataset.i, k = t.dataset.k;
  if (!k || !S.lesions[i]) return;
  const v = Number(t.value);
  if (!Number.isFinite(v)) return;
  if (k.startsWith("c")) S.lesions[i].center[+k.slice(1)] = v;
  else if (k === "radius") S.lesions[i].radius = Math.max(0.5, v);
  else if (k === "permeability") S.lesions[i].permeability = Math.max(0.001, v);
  else S.lesions[i].density = Math.min(20, Math.max(0, v));
  evaluate();
});
$("lesion-list").addEventListener("click", (e) => {
  const b = e.target.closest(".lesion-del");
  if (!b) return;
  S.lesions.splice(+b.dataset.i, 1);
  renderLesions(); evaluate();
});
$("solve").addEventListener("change", () => evaluate());
stepper("gen-dec", "gen-inc", "generations", 1);
stepper("steps-dec", "steps-inc", "fill-steps", 10);
stepper("term-dec", "term-inc", "n-terminals", 10);
$("btn-reseed").addEventListener("click", () => { $("seed").value = num("seed") + 1; evaluate(); });
$("btn-reset").addEventListener("click", () => location.reload());

// ── animate: replay growth generation by generation, then fill step by step ────
let animating = false;
async function animate() {
  if (animating) { animating = false; return; }
  animating = true;
  const btn = $("btn-animate"); btn.textContent = "■ Stop";
  const G = num("generations"), fill = $("fill").value;
  const frames = [];
  for (let g = 1; g <= G; g++) frames.push({ generations: g, fill: "none" });
  if (fill === "colonization") {
    const N = num("fill-steps");
    for (let k = 1; k <= 12; k++) frames.push({ fill_steps: Math.round(N * k / 12) });
  } else if (fill === "cco") {
    const N = num("n-terminals");
    for (let k = 1; k <= 12; k++) frames.push({ n_terminals: Math.max(1, Math.round(N * k / 12)) });
  }
  try {
    for (const f of frames) {
      if (!animating) break;
      setData(await post("/explore/grow", request({ ...f, solve: false })));
      showStats();
      status(f.generations ? `generation ${f.generations} / ${G}` : `filling: ${f.fill_steps ?? f.n_terminals}`, "ok");
      await new Promise((r) => setTimeout(r, 260));
    }
  } catch (e) { status(e.message, "err"); }
  animating = false; btn.textContent = "▶ Animate growth";
  evaluate();
}
$("btn-animate").addEventListener("click", animate);

// ── send the current setup to a project (2-D → workspace, 3-D → 3-D viewer) ─────
async function toProject(tag = "") {
  const btn = $("btn-project"); btn.disabled = true;
  status("creating project…");
  try {
    const shape = S.dim === 3 ? $("shape3").value : (S.polygon ? "custom" : $("shape2").value);
    const name = `${tag ? tag + " " : ""}explorer ${shape} (${S.dim}-D)`;
    const made = await post("/explore/project", { ...request({ solve: false }), name });
    status("project created", "ok");
    window.open(made.open, "_blank");
  } catch (e) { status(e.message, "err"); }
  finally { btn.disabled = false; }
}
$("btn-project").addEventListener("click", () => toProject());

// ── optimise the current setup (organ, roots, fill; 2-D or 3-D) ────────────────
async function optimise() {
  const btn = $("btn-opt"), label = btn.textContent;
  const free = [...document.querySelectorAll(".optfree input:checked")]
    .map((c) => c.value).filter((v) => v !== "twist" || S.dim === 3);
  if (!free.length) { $("opt-info").textContent = "tick at least one parameter to optimise"; return; }
  btn.disabled = true; btn.textContent = "Optimising…";
  $("btn-opt-use").hidden = true;
  $("opt-info").textContent = "searching… (each step grows + solves the trees)";
  status("optimising…");
  try {
    const maxiter = Math.min(200, Math.max(1, Math.round(num("maxiter")) || 15));
    $("maxiter").value = maxiter;
    const o = await post("/explore/optimize", {
      ...request({ solve: true }), free, method: $("method").value, maxiter,
    });
    $("a-gamma").value = (+o.a_gamma).toFixed(3); $("v-gamma").value = (+o.v_gamma).toFixed(3);
    $("ltr").value = o.length_to_radius; $("exponent").value = o.exponent;
    $("v-radius").value = (+o.v_radius).toFixed(3);
    if (S.dim === 3) $("twist").value = o.twist;
    const pct = (o.improvement * 100).toFixed(1);
    $("opt-info").textContent = o.kept_original
      ? `${o.kind}: no better setup at this resolution (${o.n_eval} solves) — kept yours. `
        + "Loaded into the controls."
      : `${o.kind}: ${fmt(o.start)} → ${fmt(o.best)} (${pct}% better at this resolution, `
        + `${o.n_eval} solves). Loaded into the controls.`;
    $("btn-opt-use").hidden = false;
    await evaluate();
    status("optimised", "ok");
  } catch (e) {
    $("opt-info").textContent = "optimise failed — " + e.message;
    status("optimise failed", "err");
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}
$("btn-opt").addEventListener("click", optimise);
$("btn-opt-use").addEventListener("click", () => toProject("optimised"));

// shape dropdowns come from the organ library (GET /shapes)
async function loadShapes() {
  let sh = null;
  try { sh = await (await fetch("/shapes")).json(); } catch { /* fall back below */ }
  sh = sh || { "2d": { ellipse: "ellipse" }, "3d": { ellipsoid: "ellipsoid" } };
  const opts = (m) => Object.entries(m).map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
  $("shape2").innerHTML = opts(sh["2d"]) + `<option value="custom">custom (edited outline)</option>`;
  $("shape3").innerHTML = opts(sh["3d"]);
}

// ── boot ──────────────────────────────────────────────────────────────────────
S.vp = createViewport(canvas, {
  domain: () => (S.dim === 3 ? V3 : (S.data ? S.data.domain.slice(0, 4) : [-70, 70, 0, 160])),
  // a plain drag on empty space pans in 2-D (handles still edit); 3-D drag rotates
  panWithLeft: (e) => S.dim === 2 && !!S.data && !hit2d(...S.vp.eventXY(e)),
  onChange: () => draw(),
});
loadShapes().then(() => setDim(2));          // ×4 defaults: roots, calibre, resolution
