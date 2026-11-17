// perfusion — tree explorer. Grow an arterial + venous tree into a tissue shape
// (2-D or 3-D) from user-placed roots: a fractal skeleton of N generations, then an
// optional space-filling growth (colonisation steps / CCO terminals). Every change
// re-grows (and optionally re-solves) via POST /explore/grow. Self-contained.
import { VIRIDIS, viridis, viridisCss } from "./palette.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;

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
  polygon: null,          // custom 2-D outline (null = preset)
  data: null,             // last /explore/grow response
  sel: null,              // selected outline vertex index
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
  const root = S.dim === 3 ? [num(prefix + "-x"), num(prefix + "-y"), num(prefix + "-z")]
                           : [num(prefix + "-x"), num(prefix + "-y")];
  const t = { root, radius: num(prefix + "-radius"), gamma: num(prefix + "-gamma"),
              branch_angle: num("bangle") };
  if (S.dim === 3) t.direction = dirOf(prefix, root);
  else t.angle = num(prefix + "-angle");
  return t;
}

function request(over = {}) {
  const d3 = S.dim === 3;
  const shape = d3 ? $("shape3").value : (S.polygon ? "custom" : $("shape2").value);
  return {
    dim: S.dim, shape, polygon: !d3 && S.polygon ? S.polygon : null, size: num("size"),
    artery: treeReq("a"), vein: treeReq("v"),
    generations: num("generations"), length_to_radius: num("ltr"), exponent: num("exponent"),
    fill: $("fill").value, fill_target: $("fill-target").value,
    fill_steps: num("fill-steps"), n_points: num("n-points"),
    step: optNum("step-len"), influence: optNum("influence"),
    n_terminals: num("n-terminals"), seed: num("seed"),
    solve: $("solve").checked, field: $("field").value, kind: $("kind").value,
    nx: d3 ? 22 : 60, ny: 70, ...over,
  };
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" },
                                body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = (await r.text()).slice(0, 200);
    try { const j = JSON.parse(msg); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* raw */ }
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
    S.data = await post("/explore/grow", request(over));
    draw(); showStats();
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

// ── drawing ───────────────────────────────────────────────────────────────────
function draw() {
  ctx.clearRect(0, 0, W, H);
  if (!S.data) return;
  if (S.data.dim === 3) draw3d(); else draw2d();
}

// 2-D: domain → canvas
function tc(x, y) {
  const [x0, x1, y0, y1] = S.data.domain;
  return [(x - x0) / (x1 - x0) * W, H - (y - y0) / (y1 - y0) * H];
}
function td(px, py) {
  const [x0, x1, y0, y1] = S.data.domain;
  return [x0 + px / W * (x1 - x0), y0 + (H - py) / H * (y1 - y0)];
}

const HANDLE_LEN = 14;   // domain units from root to its direction handle

function dirHandle(prefix) {
  const r = [num(prefix + "-x"), num(prefix + "-y")], a = num(prefix + "-angle") * Math.PI / 180;
  return [r[0] + HANDLE_LEN * Math.cos(a), r[1] + HANDLE_LEN * Math.sin(a)];
}

function draw2d() {
  const d = S.data;
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
    ctx.lineWidth = Math.max(0.6, r * 0.9);
    ctx.beginPath(); ctx.moveTo(p0x, p0y); ctx.lineTo(p1x, p1y); ctx.stroke();
  }
  // outline vertex handles
  d.outline.forEach(([x, y], i) => {
    const [px, py] = tc(x, y), s = S.sel === i ? 5 : 3.5;
    ctx.fillStyle = S.sel === i ? "#2563eb" : "#fff";
    ctx.strokeStyle = S.sel === i ? "#2563eb" : "#047857"; ctx.lineWidth = 1.4;
    ctx.fillRect(px - s, py - s, 2 * s, 2 * s); ctx.strokeRect(px - s, py - s, 2 * s, 2 * s);
  });
  // roots + direction handles
  rootMarker("a", true, "#dc2626"); rootMarker("v", false, "#2563eb");
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
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  updateColorbar(vmin, vmax, S.data.field.name);
}

// 3-D: orthographic projection, yaw about z then pitch about x; z is up on screen
function project(x, y, z) {
  const [x0, x1, y0, y1, z0, z1] = S.data.domain;
  let dx = x - (x0 + x1) / 2, dy = y - (y0 + y1) / 2, dz = z - (z0 + z1) / 2;
  const cy = Math.cos(S.view.yaw), sy = Math.sin(S.view.yaw);
  [dx, dy] = [dx * cy - dy * sy, dx * sy + dy * cy];
  const cp = Math.cos(S.view.pitch), sp = Math.sin(S.view.pitch);
  [dy, dz] = [dy * cp - dz * sp, dy * sp + dz * cp];
  const s = (Math.min(W, H) * 0.5 / Math.max(x1 - x0, y1 - y0, z1 - z0)) * S.view.zoom;
  return [W / 2 + dx * s, H / 2 - dz * s, dy];
}

function draw3d() {
  const d = S.data;
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, W, H);
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
    for (const p of pts) { ctx.fillStyle = viridisCss(p.t); const r = 2.4 * S.view.zoom; ctx.fillRect(p.sx - r, p.sy - r, 2 * r, 2 * r); }
    ctx.globalAlpha = 1;
    updateColorbar(min, max, "pressure");
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
    ctx.lineWidth = Math.max(0.8, s.r * 0.9 * S.view.zoom);
    ctx.beginPath(); ctx.moveTo(s.a[0], s.a[1]); ctx.lineTo(s.b[0], s.b[1]); ctx.stroke();
  }
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
    + stat(d.active_cells, "tissue cells");
  if (d.inflow != null) h += stat(fmt(d.inflow), "perfused inflow");
  if (d.objective != null) h += stat(fmt(d.objective), d.kind);
  $("summary").innerHTML = h;
  $("obj-info").textContent = d.objective != null ? `${d.kind} = ${fmt(d.objective)}`
    : ($("solve").checked ? "—" : "solving is off");
}

// ── 2-D canvas interaction: roots, aim handles, outline points ─────────────────
const canvasXY = (e) => {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * (W / r.width), (e.clientY - r.top) * (H / r.height)];
};
let drag = null;     // {what: "a-root"|"v-root"|"a-aim"|"v-aim"|"vertex"|"rotate", i, lx, ly}

function hit2d(px, py) {
  const near = (x, y, tol) => { const [cx, cy] = tc(x, y); return (cx - px) ** 2 + (cy - py) ** 2 < tol * tol; };
  for (const p of ["a", "v"]) {
    if (near(...dirHandle(p), 9)) return { what: p + "-aim" };
    if (near(num(p + "-x"), num(p + "-y"), 11)) return { what: p + "-root" };
  }
  const o = S.data.outline;
  for (let i = 0; i < o.length; i++) if (near(o[i][0], o[i][1], 8)) return { what: "vertex", i };
  return null;
}

function ensureCustom() {
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
  if (drag.what.endsWith("-root")) {
    const p = drag.what[0];
    $(p + "-x").value = x.toFixed(1); $(p + "-y").value = y.toFixed(1);
  } else if (drag.what.endsWith("-aim")) {
    const p = drag.what[0];
    const ang = Math.atan2(y - num(p + "-y"), x - num(p + "-x")) * 180 / Math.PI;
    $(p + "-angle").value = Math.round(ang);
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

canvas.addEventListener("wheel", (e) => {
  if (!S.data || S.data.dim !== 3) return;
  e.preventDefault();
  S.view.zoom = Math.max(0.4, Math.min(4, S.view.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
  draw();
}, { passive: false });

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
const LABELS = { size: [2], generations: [0], ltr: [1], exponent: [2], bangle: [0],
                 "fill-steps": [0], "n-terminals": [0] };
function syncLabels() {
  for (const [id, [dp]] of Object.entries(LABELS)) {
    const el = $("v-" + id);
    if (el) el.textContent = num(id).toFixed(dp);
  }
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
  const D = DEFAULTS[dim];
  for (const [p, t] of [["a", D.art], ["v", D.ven]]) {
    $(p + "-x").value = t.root[0]; $(p + "-y").value = t.root[1];
    if (dim === 3) { $(p + "-z").value = t.root[2]; $(p + "-dir").value = t.dir; }
    else $(p + "-angle").value = t.angle;
  }
}

function setDim(dim) {
  S.dim = dim; S.sel = null; S.data = null;
  setRoots(dim);
  // scale the skeleton to the domain: the 3-D box (60³) is smaller than the 2-D
  // window (140×160), and 3-D colonisation grows many segments per step
  const D = dim === 3 ? { ltr: 5, gen: 4, ar: 1.5, vr: 1.8, steps: 25 }
                      : { ltr: 10, gen: 4, ar: 2, vr: 2.4, steps: 40 };
  $("ltr").value = D.ltr; $("generations").value = D.gen; $("fill-steps").value = D.steps;
  $("a-radius").value = D.ar; $("v-radius").value = D.vr;
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
  if ($("shape2").value !== "custom") S.polygon = null;     // back to a preset
  else if (!S.polygon && S.data) S.polygon = S.data.outline.map((p) => [...p]);
  S.sel = null; evaluate();
});
$("size").addEventListener("input", () => {
  if (S.polygon) { S.polygon = null; $("shape2").value = "ellipse"; }   // size rescales presets
  syncLabels(); schedule();
});
// sliders re-grow while dragging (debounced); selects / numbers on change
for (const id of ["generations", "ltr", "exponent", "bangle", "fill-steps", "n-terminals"]) {
  $(id).addEventListener("input", () => { syncLabels(); schedule(); });
}
for (const id of ["shape3", "fill", "fill-target", "field", "kind", "a-dir", "v-dir"]) {
  $(id).addEventListener("change", () => { syncVisibility(); evaluate(); });
}
for (const id of ["a-x", "a-y", "a-z", "a-angle", "a-radius", "a-gamma",
                  "v-x", "v-y", "v-z", "v-angle", "v-radius", "v-gamma",
                  "n-points", "step-len", "influence", "seed"]) {
  $(id).addEventListener("change", () => evaluate());
}
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
      S.data = await post("/explore/grow", request({ ...f, solve: false }));
      draw(); showStats();
      status(f.generations ? `generation ${f.generations} / ${G}` : `filling: ${f.fill_steps ?? f.n_terminals}`, "ok");
      await new Promise((r) => setTimeout(r, 260));
    }
  } catch (e) { status(e.message, "err"); }
  animating = false; btn.textContent = "▶ Animate growth";
  evaluate();
}
$("btn-animate").addEventListener("click", animate);

// ── optimise (2-D default frame): loads the best asymmetry / L:R / exponent ────
async function optimise() {
  const btn = $("btn-opt"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "Optimising…";
  $("opt-info").textContent = "searching…";
  status("optimising… (this runs several solves)");
  try {
    const o = await post("/explore/optimize", {
      kind: $("kind").value, method: $("method").value,
      maxiter: num("maxiter") || 15, nx: 50, ny: 58,
    });
    $("a-gamma").value = (+o.a_gamma).toFixed(2); $("v-gamma").value = (+o.v_gamma).toFixed(2);
    $("ltr").value = o.length_to_radius; $("exponent").value = o.exponent;
    $("opt-info").textContent =
      `${o.kind}: ${fmt(o.start)} → ${fmt(o.best)} (${(o.improvement * 100).toFixed(1)}% better, ${o.n_eval} evals)`;
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

// ── boot ──────────────────────────────────────────────────────────────────────
setRoots(2);
syncVisibility(); syncLabels();
evaluate();
