// perfusion — dependency-free 3-D viewer (canvas orthographic projection).
// Renders the coupled 3-D hybrid result (vessel trees + pressure voxel field)
// from GET /demo3d. No WebGL / vtk.js, so it works offline and is easy to verify.

import { viridisCss } from "./palette.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view3d");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;

const view = { yaw: 0.7, pitch: 0.45, zoom: 1, spin: true };
let data = null, center = [0, 0, 0], scale = 1;

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
  const r = await fetch(`/projects/${PROJECT}/view3d`);
  if (!r.ok) {
    $("status").textContent = r.status === 404 ? "project not found" : "3-D view unavailable";
    $("status").className = "status err"; return;
  }
  setData(await r.json());
  $("status").textContent = data.has_result
    ? "showing solved result" : "showing vessels — run a simulation for the 3-D field";
  $("status").className = "status ok";
}

function setData(d) {
  data = d;
  const [x0, x1, y0, y1, z0, z1] = data.domain;
  center = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
  const span = Math.max(x1 - x0, y1 - y0, z1 - z0);
  scale = (Math.min(W, H) * 0.42) / span;
  $("info").textContent = `${data.segments.length} segments · ${data.pressure.voxels.length} field points · grid ${data.shape.join("×")}`;
  if (!PROJECT) { $("status").textContent = "ready"; $("status").className = "status ok"; }
}

// Rotate a centred point by yaw (about z / vertical) then pitch (about x),
// orthographic project. Returns [screenX, screenY, depth].
function project(x, y, z) {
  let dx = x - center[0], dy = y - center[1], dz = z - center[2];
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  [dx, dy] = [dx * cy - dy * sy, dx * sy + dy * cy];
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  [dy, dz] = [dy * cp - dz * sp, dy * sp + dz * cp];
  const s = scale * view.zoom;
  return [W / 2 + dx * s, H / 2 - dz * s, dy];    // z is "up" on screen; depth = dy
}

function render() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, W, H);
  if (!data) return;

  if ($("show-field").checked) {
    const { min, max, voxels } = data.pressure;
    const span = (max - min) || 1;
    const alpha = Number($("opacity").value) / 100;
    const pts = voxels.map((v) => {
      const [sx, sy, depth] = project(v[0], v[1], v[2]);
      return { sx, sy, depth, t: (v[3] - min) / span };
    }).sort((a, b) => a.depth - b.depth);          // painter's: far first
    ctx.globalAlpha = alpha;
    for (const p of pts) {
      ctx.fillStyle = viridisCss(p.t);
      const r = 2.6 * view.zoom;
      ctx.fillRect(p.sx - r, p.sy - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
  }

  if ($("show-trees").checked) {
    for (const s of data.segments) {
      const [ax, ay] = project(s[0], s[1], s[2]);
      const [bx, by] = project(s[3], s[4], s[5]);
      ctx.strokeStyle = s[7] === 0 ? "#ff6b6b" : "#5b8dff";   // artery / vein
      ctx.lineWidth = Math.max(1, s[6] * 0.9 * view.zoom);
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
canvas.addEventListener("pointerdown", (e) => { dragging = true; view.spin = false; $("spin").checked = false; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("pointerup", () => { dragging = false; });
window.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  view.yaw += (e.clientX - lastX) * 0.01;
  view.pitch += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  view.zoom = Math.max(0.3, Math.min(4, view.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
}, { passive: false });
$("spin").addEventListener("change", (e) => { view.spin = e.target.checked; });
$("reset").addEventListener("click", () => { Object.assign(view, { yaw: 0.7, pitch: 0.45, zoom: 1 }); });
$("gen").addEventListener("change", () => load().catch(() => {}));
$("model").addEventListener("change", () => load().catch(() => {}));

const initModel = new URLSearchParams(location.search).get("model");
if (initModel === "frog" || initModel === "synthetic") $("model").value = initModel;
load().catch((e) => { $("status").textContent = e.message; });
loop();
