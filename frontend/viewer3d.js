// perfusion — dependency-free 3-D viewer (canvas orthographic projection).
// Renders the coupled 3-D hybrid result (vessel trees + pressure voxel field)
// from GET /demo3d. No WebGL / vtk.js, so it works offline and is easy to verify.

import { viridisCss } from "./palette.js";
import { createViewport } from "./viewport.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view3d");
const ctx = canvas.getContext("2d");
// points are projected to normalised coords (±0.5 spans the scene), then the shared
// viewport maps them to the canvas: it fills the stage, is HiDPI-crisp, and gives
// scroll-zoom, right-drag / Space+drag pan and ＋ − ⤢ buttons.
const vp = createViewport(canvas, { domain: () => [-0.62, 0.62, -0.62, 0.62] });

const view = { yaw: 0.7, pitch: 0.45, spin: true };
let data = null, center = [0, 0, 0], span = 1;

// If arrived from a project (?project=<id>) keep working on it: show ITS result in
// 3-D. Otherwise fall back to the standalone demos (synthetic / frog).
const PROJECT = new URLSearchParams(location.search).get("project");
if (PROJECT) $("link-2d").href = `/?project=${PROJECT}`;   // toggle back keeps the project

async function load() {
  if (PROJECT) return loadProject();
  const model = $("model").value;
  const gen = $("gen").value;
  $("gen-row").style.display = model === "synthetic" ? "" : "none";
  $("status").textContent = model === "frog" ? "solving frog hybrid…" : "solving 3-D hybrid…";
  const r = await fetch(`/demo3d?model=${model}&gen=${gen}`);
  if (!r.ok) {
    const msg = r.status === 501 ? "frog needs the image extra" : "3-D endpoint unavailable";
    $("status").textContent = msg; $("status").className = "status err"; return;
  }
  setData(await r.json());
}

async function loadProject() {
  $("model-row").style.display = "none";     // project mode: no demo picker
  $("gen-row").style.display = "none";
  $("status").textContent = "loading project…";
  const r = await fetch(`/projects/${PROJECT}/view3d?field=${$("v3-field").value}`);
  if (!r.ok) {
    $("status").textContent = r.status === 404 ? "project not found" : "3-D view unavailable";
    $("status").className = "status err"; return;
  }
  setData(await r.json());
  if (data.dim === 3) show3dProject();
  $("status").textContent = data.has_result
    ? "showing solved result" : "showing vessels — run a simulation for the 3-D field";
  $("status").className = "status ok";
}

// A 3-D project (e.g. sent from the Tree explorer): run it here and pick the field.
function show3dProject() {
  $("proj-card").hidden = false;
  $("link-2d").href = "/explorer.html";          // 3-D projects live in the explorer/viewer
  $("link-2d").textContent = "🌳 Tree explorer";
  $("proj-desc").textContent = `${data.tissue || "tissue"} · grid ${data.shape.join("×")} · `
    + `${data.segments.length} vessel segments`;
  const s = data.summary;
  const f = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
  $("v3-summary").innerHTML = s && data.has_result
    ? [["arterial inflow", s.arterial_inflow], ["venous outflow", s.venous_outflow],
       ["MTT", s.mtt], ["CTH", s.cth]]
      .map(([k, v]) => `<div class="stat"><b>${f(v)}</b><span>${k}</span></div>`).join("")
    : "";
  $("v3-run").textContent = data.has_result ? "↻ Re-run 3-D simulation" : "▶ Run 3-D simulation";
}

async function run3d() {
  const btn = $("v3-run"); btn.disabled = true;
  $("v3-run-info").textContent = "solving…";
  try {
    const job = await (await fetch(`/projects/${PROJECT}/run`, {
      method: "POST", headers: { "content-type": "application/json" }, body: "{}" })).json();
    let st = job;
    while (st.state !== "done" && st.state !== "error") {
      await new Promise((r) => setTimeout(r, 400));
      st = await (await fetch(`/jobs/${job.id}`)).json();
    }
    $("v3-run-info").textContent = st.state === "done" ? "" : (st.error || "run failed");
    await loadProject();
  } finally { btn.disabled = false; }
}
$("v3-run").addEventListener("click", () => run3d().catch((e) => { $("v3-run-info").textContent = e.message; }));
$("v3-field").addEventListener("change", () => loadProject().catch(() => {}));

function setData(d) {
  data = d;
  const [x0, x1, y0, y1, z0, z1] = data.domain;
  center = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
  span = Math.max(x1 - x0, y1 - y0, z1 - z0);
  $("info").textContent = `${data.segments.length} segments · ${data.pressure.voxels.length} field points · grid ${data.shape.join("×")}`;
  if (!PROJECT) { $("status").textContent = "ready"; $("status").className = "status ok"; }
}

// Rotate a centred point by yaw (about z / vertical) then pitch (about x),
// orthographic project. Returns [screenX, screenY, depth].
function project(x, y, z) {
  let dx = (x - center[0]) / span, dy = (y - center[1]) / span, dz = (z - center[2]) / span;
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  [dx, dy] = [dx * cy - dy * sy, dx * sy + dy * cy];
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  [dy, dz] = [dy * cp - dz * sp, dy * sp + dz * cp];
  const [sx, sy2] = vp.toCanvas(dx, dz);        // z is "up" on screen; depth = dy
  return [sx, sy2, dy];
}

function render() {
  vp.begin();
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, vp.W, vp.H);
  const zs = vp.scale / 500;                      // size factor that follows the zoom
  if (!data) return;

  if ($("show-field").checked) {
    const { min, max, voxels } = data.pressure;
    const span = (max - min) || 1;
    const alpha = Number($("opacity").value) / 100;
    const pts = voxels.map((v) => {
      const [sx, sy, depth] = project(v[0], v[1], v[2]);
      return { sx, sy, depth, t: Math.max(0, Math.min(1, (v[3] - min) / span)) };
    }).sort((a, b) => a.depth - b.depth);          // painter's: far first
    ctx.globalAlpha = alpha;
    for (const p of pts) {
      ctx.fillStyle = viridisCss(p.t);
      const r = Math.max(1.2, 2.6 * zs);
      ctx.fillRect(p.sx - r, p.sy - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
  }

  if (!data.pressure.voxels.length && data.tissue_points) {   // unsolved 3-D project
    ctx.fillStyle = "rgba(52,211,153,.35)";
    for (const v of data.tissue_points) {
      const [sx, sy] = project(v[0], v[1], v[2]);
      ctx.fillRect(sx - 1, sy - 1, 2, 2);
    }
  }

  if ($("show-trees").checked) {
    for (const s of data.segments) {
      const [ax, ay] = project(s[0], s[1], s[2]);
      const [bx, by] = project(s[3], s[4], s[5]);
      ctx.strokeStyle = s[7] === 0 ? "#ff6b6b" : "#5b8dff";   // artery / vein
      ctx.lineWidth = Math.max(1, s[6] * 0.9 * zs);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
  }
}

function loop() {
  if (view.spin && data) view.yaw += 0.006;
  render();
  requestAnimationFrame(loop);
}

// ── interaction ────────────────────────────────────────────────────────────
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return; dragging = true; view.spin = false; $("spin").checked = false; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("pointerup", () => { dragging = false; });
window.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  view.yaw += (e.clientX - lastX) * 0.01;
  view.pitch += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

$("spin").addEventListener("change", (e) => { view.spin = e.target.checked; });
$("reset").addEventListener("click", () => { Object.assign(view, { yaw: 0.7, pitch: 0.45 }); vp.fit(); });
$("gen").addEventListener("change", () => load().catch(() => {}));
$("model").addEventListener("change", () => load().catch(() => {}));

const initModel = new URLSearchParams(location.search).get("model");
if (initModel === "frog" || initModel === "synthetic") $("model").value = initModel;
load().catch((e) => { $("status").textContent = e.message; });
loop();
