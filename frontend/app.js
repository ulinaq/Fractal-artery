// perfusion — 2-D workspace front-end (no build step, ES modules).
// Talks to the FastAPI back-end that serves this file.

// ── domain / canvas geometry ───────────────────────────────────────────────
const DOMAIN = { xmin: 0, xmax: 100, ymin: 0, ymax: 120 };   // physical window
const GRID_L = { lx: DOMAIN.xmax - DOMAIN.xmin, ly: DOMAIN.ymax - DOMAIN.ymin };
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const sx = W / (DOMAIN.xmax - DOMAIN.xmin);
const sy = H / (DOMAIN.ymax - DOMAIN.ymin);
const toCanvas = (x, y) => [(x - DOMAIN.xmin) * sx, H - (y - DOMAIN.ymin) * sy];
const toDomain = (px, py) => [DOMAIN.xmin + px / sx, DOMAIN.ymin + (H - py) / sy];

// ── viridis colour ramp (perceptual, colour-blind safe) ─────────────────────
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];
function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(t), f = t - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ── tiny API client ─────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["content-type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.status === 204 ? null : r.json();
}

// ── application state ────────────────────────────────────────────────────────
const state = {
  project: null,
  mode: "draw",              // "draw" | "result"
  field: "pressure",
  vessels: { arteries: [], veins: [] },   // saved strokes
  stroke: [],                // current in-progress polyline (domain coords)
  network: null,             // {segments:[...]} overlaid in result mode
  fieldData: null,           // {name, shape, data, vmin, vmax}
};

// ── status helpers ────────────────────────────────────────────────────────────
const statusEl = document.getElementById("status");
function status(text, kind = "") { statusEl.textContent = text; statusEl.className = "status " + kind; }
const $ = (id) => document.getElementById(id);
const numOf = (id) => Number($(id).value);
const vtype = () => document.querySelector('input[name=vtype]:checked').value;

// ── project ───────────────────────────────────────────────────────────────────
async function newProject() {
  const body = {
    name: $("proj-name").value || "workspace",
    params: { p_in: numOf("p_in"), p_out: numOf("p_out"),
              r_min_frac: numOf("r_min_frac"), seed: numOf("seed") },
    grid: { nx: numOf("nx"), ny: numOf("ny"), ...GRID_L,
            domain_limit: [DOMAIN.xmin, DOMAIN.xmax, DOMAIN.ymin, DOMAIN.ymax] },
  };
  const p = await api("POST", "/projects", body);
  state.project = p.id;
  state.vessels = { arteries: [], veins: [] };
  state.stroke = []; state.network = null; state.fieldData = null;
  $("proj-id").textContent = `project ${p.id}`;
  $("btn-export").disabled = true;
  $("grow-info").textContent = ""; $("run-info").textContent = "";
  setMode("draw");
  status("project ready", "ok");
  redraw();
}

function requireProject() {
  if (!state.project) { status("create a project first", "err"); throw new Error("no project"); }
  return state.project;
}

// ── vessel drawing ────────────────────────────────────────────────────────────
canvas.addEventListener("pointerdown", (e) => {
  if (state.mode !== "draw") return;
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (W / rect.width);
  const py = (e.clientY - rect.top) * (H / rect.height);
  state.stroke.push(toDomain(px, py));
  redraw();
});

function endStroke() {
  if (state.stroke.length >= 2) {
    const bucket = vtype() === "artery" ? state.vessels.arteries : state.vessels.veins;
    bucket.push({ points: state.stroke.map(([x, y]) => [x, y]), radius: numOf("draw-radius") });
  }
  state.stroke = [];
  redraw();
}
function undo() {
  if (state.stroke.length) { state.stroke.pop(); }
  else {
    const bucket = vtype() === "artery" ? state.vessels.arteries : state.vessels.veins;
    bucket.pop();
  }
  redraw();
}
function clearVessels() { state.vessels = { arteries: [], veins: [] }; state.stroke = []; redraw(); }

async function saveVessels() {
  requireProject();
  await api("PUT", `/projects/${state.project}/vessels`, state.vessels);
  $("use-vessels").checked = true;
  status(`saved ${state.vessels.arteries.length} arteries, ${state.vessels.veins.length} veins`, "ok");
}

// ── grow / run ─────────────────────────────────────────────────────────────────
async function grow() {
  requireProject();
  if (state.vessels.arteries.length) await saveVessels();
  status("growing…");
  const info = await api("POST", `/projects/${state.project}/grow`,
    { n_points: numOf("grow-npts"), step: numOf("grow-step"), seed: numOf("seed") });
  $("grow-info").textContent = `grown tree: ${info.n_segments} segments, ${info.n_terminals} terminals`;
  await loadNetwork();
  status("arterial tree grown", "ok");
  redraw();
}

async function run() {
  requireProject();
  if (state.vessels.arteries.length || state.vessels.veins.length) await saveVessels();
  const job = await api("POST", `/projects/${state.project}/run`,
    { nonlinear: $("nonlinear").checked, use_vessels: $("use-vessels").checked });
  status("running…");
  $("run-info").textContent = "job " + job.id;
  const final = await pollJob(job.id);
  if (final.state === "error") { status("run failed", "err"); $("run-info").textContent = final.error; return; }
  status("solved", "ok");
  await Promise.all([loadNetwork(), showSummary(), loadField(state.field)]);
  $("btn-export").disabled = false;
  setMode("result");
}

async function pollJob(jid) {
  for (;;) {
    const s = await api("GET", `/jobs/${jid}`);
    if (s.state === "done" || s.state === "error") return s;
    await new Promise((r) => setTimeout(r, 300));
  }
}

async function showSummary() {
  const s = await api("GET", `/projects/${state.project}/results`);
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
  $("summary").innerHTML = [
    ["arterial inflow", s.arterial_inflow], ["venous outflow", s.venous_outflow],
    ["pressure max", s.pressure_max], ["flux mean", s.flux_mean],
  ].map(([k, v]) => `<div class="stat"><b>${fmt(v)}</b><span>${k}</span></div>`).join("");
}

async function loadNetwork() {
  try { state.network = await api("GET", `/projects/${state.project}/network`); }
  catch { state.network = null; }
}

async function loadField(name) {
  state.field = name;
  const f = await api("GET", `/projects/${state.project}/results/field/${name}`);
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  state.fieldData = { ...f, vmin, vmax };
  redraw();
}

// ── rendering ──────────────────────────────────────────────────────────────────
function redraw() {
  ctx.clearRect(0, 0, W, H);
  if (state.mode === "result" && state.fieldData) drawField();
  drawNetworkOverlay();
  drawVessels();
  drawStroke();
}

function drawField() {
  const { shape, data, vmin, vmax } = state.fieldData;
  const [nx, ny] = shape;
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(nx, ny);
  const span = vmax - vmin || 1;
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      const v = data[ix][iy];
      const px = ix, py = ny - 1 - iy;              // flip y (domain up → canvas up)
      const o = (py * nx + px) * 4;
      if (v === null) { img.data[o + 3] = 0; continue; }
      const [r, g, b] = viridis((v - vmin) / span);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  updateColorbar(vmin, vmax);
}

function drawNetworkOverlay() {
  if (state.mode !== "result" || !state.network) return;
  for (const s of state.network.segments) {
    const [x0, y0] = toCanvas(s.x0, s.y0), [x1, y1] = toCanvas(s.x1, s.y1);
    ctx.strokeStyle = s.kind === "vein" ? "rgba(191,219,254,.95)" : "rgba(254,202,202,.95)";
    ctx.lineWidth = Math.max(0.6, s.r * 0.5);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
}

function drawVessels() {
  const paint = (list, color) => {
    ctx.strokeStyle = color; ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const v of list) {
      ctx.lineWidth = Math.max(1.5, v.radius * 1.2);
      ctx.beginPath();
      v.points.forEach(([x, y], i) => { const [px, py] = toCanvas(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    }
  };
  paint(state.vessels.arteries, getCss("--art"));
  paint(state.vessels.veins, getCss("--ven"));
}

function drawStroke() {
  if (state.mode !== "draw" || !state.stroke.length) return;
  ctx.strokeStyle = vtype() === "artery" ? getCss("--art") : getCss("--ven");
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = Math.max(1.5, numOf("draw-radius") * 1.2);
  ctx.beginPath();
  state.stroke.forEach(([x, y], i) => { const [px, py] = toCanvas(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();
  for (const [x, y] of state.stroke) { const [px, py] = toCanvas(x, y); ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill(); }
}

function updateColorbar(vmin, vmax) {
  const cb = $("colorbar"); cb.hidden = false;
  const stops = VIRIDIS.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i / (VIRIDIS.length - 1)).toFixed(0)}%`);
  $("cb-ramp").style.background = `linear-gradient(to top, ${stops.join(",")})`;
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(1) : v.toFixed(2);
  $("cb-max").textContent = fmt(vmax); $("cb-min").textContent = fmt(vmin);
  $("cb-title").textContent = state.field;
}

const getCss = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ── modes ────────────────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  $("mode-draw").classList.toggle("active", mode === "draw");
  $("mode-result").classList.toggle("active", mode === "result");
  $("field-picker").classList.toggle("show", mode === "result");
  $("colorbar").hidden = mode !== "result" || !state.fieldData;
  canvas.style.cursor = mode === "draw" ? "crosshair" : "default";
  redraw();
}

// ── wiring ────────────────────────────────────────────────────────────────────
$("btn-new").addEventListener("click", () => newProject().catch((e) => status(e.message, "err")));
$("btn-endseg").addEventListener("click", endStroke);
$("btn-undo").addEventListener("click", undo);
$("btn-clear").addEventListener("click", clearVessels);
$("btn-save-vessels").addEventListener("click", () => saveVessels().catch((e) => status(e.message, "err")));
$("btn-grow").addEventListener("click", () => grow().catch((e) => status(e.message, "err")));
$("btn-run").addEventListener("click", () => run().catch((e) => status(e.message, "err")));
$("btn-export").addEventListener("click", () => { window.location = `/projects/${state.project}/export`; });
$("mode-draw").addEventListener("click", () => setMode("draw"));
$("mode-result").addEventListener("click", () => setMode("result"));
canvas.addEventListener("dblclick", endStroke);
document.querySelectorAll(".field").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".field").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  if (state.project) loadField(b.dataset.field).catch((e) => status(e.message, "err"));
}));

// Open an existing project by id (shareable URL: ?project=<id>).
async function openProject(id) {
  const info = await api("GET", `/projects/${id}`);
  state.project = info.id;
  $("proj-name").value = info.name;
  $("proj-id").textContent = `project ${info.id}`;
  await loadNetwork();
  if (info.has_result) {
    await Promise.all([showSummary(), loadField(state.field)]);
    $("btn-export").disabled = false;
    setMode("result");
  } else { setMode("draw"); }
  status(`opened ${info.id}`, "ok");
}

// ── boot ──────────────────────────────────────────────────────────────────────
(async function boot() {
  try { const h = await api("GET", "/health"); status(`connected · v${h.version}`, "ok"); }
  catch { status("back-end unreachable", "err"); redraw(); return; }
  const shared = new URLSearchParams(location.search).get("project");
  if (shared) { try { await openProject(shared); return; } catch (e) { status(e.message, "err"); } }
  redraw();
})();
