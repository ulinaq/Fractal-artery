// perfusion — dependency-free 3-D viewer (canvas orthographic projection).
// Renders the coupled 3-D hybrid result (vessel trees + pressure voxel field)
// from GET /demo3d. No WebGL / vtk.js, so it works offline and is easy to verify.

import { viridisCss } from "./palette.js";
import { createViewport } from "./viewport.js";
import { uqHtml } from "./uqview.js";
import { createEditor3d } from "./edit3d.js";
import { createRegions3d } from "./regions3d.js";

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

const OXYGEN = new Set(["saturation", "consumption"]);
const anim = { idx: 0, playing: true, last: 0 };     // tracer (flow video) playback

async function loadProject() {
  $("model-row").style.display = "none";     // project mode: no demo picker
  $("gen-row").style.display = "none";
  const field = $("v3-field").value;
  const mc = field.startsWith("mc_");
  $("v3-k-row").hidden = !OXYGEN.has(field);
  $("v3-mc-row").hidden = !mc;
  $("status").textContent = OXYGEN.has(field) ? "solving oxygen…"
    : mc ? "solving the compartments…" : "loading project…";
  const q = new URLSearchParams({ field, k: $("v3-k").value || 1,
                                  n: $("v3-mc-n").value || 3, beta: $("v3-mc-beta").value || 1 });
  const r = await fetch(`/projects/${PROJECT}/view3d?${q}`);
  if (r.status === 409 && field !== "pressure") {   // tracer field before the flow exists
    $("status").textContent = "compute the 🌊 flow first for tracer fields";
    $("status").className = "status err";
    $("v3-field").value = "pressure";
    return loadProject();
  }
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    $("status").textContent = r.status === 404 ? "project not found"
      : typeof j.detail === "string" ? j.detail : "3-D view unavailable";
    $("status").className = "status err"; return;
  }
  setData(await r.json());
  if (data.direct) $("v3-model").value = "direct";      // reflect the stored result's model
  if (data.tracer || data.net_tracer) startAnimation();
  else $("v3-play-row").hidden = true;
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
  const f = (v) => Number.isInteger(v) ? String(v)
    : (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
  const direct = !!data.direct;
  $("v3-rhoc-row").hidden = $("v3-model").value !== "direct";
  $("v3-field-row").hidden = direct;               // the direct model has no tissue field
  $("v3-summary").innerHTML = s && data.has_result
    ? [["arterial inflow", s.arterial_inflow], ["venous outflow", s.venous_outflow],
       ...(direct ? [["connections", s.n_connections]] : [["MTT", s.mtt], ["CTH", s.cth]])]
      .map(([k, v]) => `<div class="stat"><b>${f(v)}</b><span>${k}</span></div>`).join("")
    : "";
  if (data.oef != null) {
    $("v3-summary").innerHTML += `<div class="stat"><b>${f(data.oef)}</b><span>O₂ extraction (OEF)</span></div>`;
  }
  if (data.mc && data.mc.perfusion != null) {
    $("v3-summary").innerHTML += `<div class="stat"><b>${f(data.mc.perfusion)}</b>`
      + `<span>capillary exchange (N=${data.mc.n})</span></div>`;
  }
  if (s && data.has_result && !$("v3-cal-target").value) {   // a sensible default target
    $("v3-cal-target").value = Number((1.2 * s.arterial_inflow).toPrecision(3));
  }
  $("v3-an-card").hidden = direct;                 // UQ / calibration re-solve the hybrid
  syncExport();
  syncVideo();
  $("v3-flow-desc").textContent = direct
    ? "Time a contrast bolus along the vessels and their terminal links, then play it."
    : "Advect a contrast bolus through the 3-D bed, then play it or map its kinetics.";                 // UQ / calibration re-solve the hybrid
  $("v3-run").textContent = data.has_result ? "↻ Re-run 3-D simulation" : "▶ Run 3-D simulation";
  $("v3-flow-card").hidden = !data.has_result;    // flow needs a solved field
  $("v3-tree-card").hidden = false;
  if ($("v3-reg-card").hidden) { $("v3-reg-card").hidden = false; regions.load(); }
}

// the active bolus animation: tissue voxels (hybrid) or network elements (direct)
const tracerOf = (d) => (d && (d.tracer || d.net_tracer)) || null;

// The flow video, rendered on the server (rotating maximum-intensity projection).
function syncVideo() {
  $("v3-video-row").hidden = !(data && data.has_tracer && !data.direct);
  const q = `?spin=${$("v3-video-spin").checked}`;
  $("v3-gif").href = `/projects/${PROJECT}/tracer.gif${q}`;
  $("v3-mp4").href = `/projects/${PROJECT}/tracer.mp4${q}`;
}

// Export: field formats need a solved hybrid field; the vessels (.vtp) always export.
function syncExport() {
  const fmt = $("v3-export-fmt");
  const hasField = !!(data && data.has_result && !data.direct);
  for (const o of fmt.options) o.disabled = o.value !== "vtp" && !hasField;
  if (fmt.selectedOptions[0].disabled) { fmt.value = "vtp"; fmt.dataset.forced = "1"; }
  else if (hasField && fmt.dataset.forced) { fmt.value = "npz"; delete fmt.dataset.forced; }
  $("v3-bundle").disabled = !hasField;
}

function startAnimation() {
  const t = tracerOf(data);
  $("v3-play-row").hidden = false;
  $("v3-scrub").max = t.frames.length - 1;
  anim.idx = Math.min(anim.idx, t.frames.length - 1);
  showFrame();
}

function showFrame() {
  const t = tracerOf(data);
  if (!t) return;
  $("v3-scrub").value = anim.idx;
  $("v3-time").textContent = `t = ${t.times[anim.idx].toFixed(2)}`;
}

// POST a job (run / tracer) and wait for it; returns the final job record.
async function runJob(path, body = {}) {
  const job = await (await fetch(`/projects/${PROJECT}/${path}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).json();
  let st = job;
  while (st.state !== "done" && st.state !== "error") {
    await new Promise((r) => setTimeout(r, 400));
    st = await (await fetch(`/jobs/${job.id}`)).json();
  }
  return st;
}

async function postJSON(path, body) {
  const r = await fetch(`/projects/${PROJECT}/${path}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : `${path} failed (${r.status})`);
  return j;
}

async function run3d() {
  const btn = $("v3-run"); btn.disabled = true;
  $("v3-run-info").textContent = "solving…";
  try {
    const model = $("v3-model").value;
    const st = await runJob("run", { model, rho_c: Number($("v3-rhoc").value) || 0 });
    $("v3-run-info").textContent = st.state === "done" ? "" : (st.error || "run failed");
    $("v3-kin").textContent = "";
    if (model === "direct" || !["pressure", "flux", "tof"].includes($("v3-field").value)) {
      $("v3-field").value = "pressure";
    }
    await loadProject();
  } finally { btn.disabled = false; }
}

async function flow3d() {
  const btn = $("v3-flow"); btn.disabled = true;
  $("v3-kin").textContent = data && data.direct ? "timing the bolus along the network…"
    : "advecting the bolus through the 3-D bed…";
  try {
    const st = await runJob("tracer", { n_frames: 40 });
    if (st.state !== "done") { $("v3-kin").textContent = st.error || "flow failed"; return; }
    $("v3-field").value = "tracer";
    anim.idx = 0; anim.playing = true; $("v3-play").textContent = "⏸";
    await loadProject();
    const f = (v) => (v == null ? "–" : Math.abs(v) < 1e-3 || Math.abs(v) > 1e4 ? v.toExponential(2) : v.toFixed(3));
    if (data.net_tracer) {            // direct model: transit times along the network
      const arr = data.net_tracer.arrivals.filter((a) => a != null);
      $("v3-kin").textContent = `bolus timed along ${data.net_tracer.lines.length} vessels + links`
        + ` · reaches ${Math.round(100 * arr.length / data.net_tracer.arrivals.length)}%`
        + ` · last arrival t = ${f(Math.max(...arr, 0))}`;
      return;
    }
    const k = await (await fetch(`/projects/${PROJECT}/kinetics?model=1CM`)).json();
    $("v3-kin").textContent = `1-compartment fit of the tissue curve: Fp = ${f(k.Fp)}, vp = ${f(k.vp)}`
      + " · pick Fp / vp / CBV / MTT for voxel maps";
  } finally { btn.disabled = false; }
}

// grow / prune / adapt the 3-D trees, then re-show (the old result is invalidated)
async function treeOp(kind) {
  const info = $("v3-tree-info");
  const target = $("v3-target").value;
  ["v3-grow", "v3-prune", "v3-adapt"].forEach((id) => ($(id).disabled = true));
  info.textContent = { grow: "growing…", prune: "pruning…", adapt: "remodelling radii (solves repeatedly)…" }[kind];
  try {
    let msg;
    if (kind === "grow") {
      const r = await postJSON("grow", { target, method: $("v3-method").value,
                                         n_points: Number($("v3-npts").value) || 60 });
      msg = `${target}: ${r.n_segments} segments, ${r.n_terminals} terminals`;
    } else if (kind === "prune") {
      const r = await postJSON("prune", { target, steps: 1 });
      msg = `${target}: removed ${r.n_removed} segments → ${r.n_segments} left`;
    } else {
      const r = await postJSON("adapt", {});
      msg = `Murray exponent ${r.murray_start.toFixed(2)} → ${r.murray_final.toFixed(2)} in ${r.n_iter} steps`;
    }
    info.textContent = msg + " · re-run to solve the new trees";
    if (editor && editor.active) await editor.setActive(true);   // edit the new trees
    $("v3-field").value = "pressure"; $("v3-kin").textContent = "";
    await loadProject();
  } catch (e) {
    info.textContent = e.message;
  } finally {
    ["v3-grow", "v3-prune", "v3-adapt"].forEach((id) => ($(id).disabled = false));
  }
}
// Monte-Carlo / Sobol uncertainty over the ticked input ranges (each sample a 3-D solve)
async function uq3d() {
  const params = {};
  const range = (id) => [Number($(`${id}-lo`).value), Number($(`${id}-hi`).value)];
  if ($("v3-uq-pin").checked) params.p_in = range("v3-uq-pin");
  if ($("v3-uq-mu").checked) params.mu = range("v3-uq-mu");
  if ($("v3-uq-k").checked) params.permeability = range("v3-uq-k");
  const out = $("v3-uq-out");
  if (!Object.keys(params).length) { out.textContent = "tick at least one uncertain input"; return; }
  const sobol = $("v3-uq-sobol").checked;
  const n = Math.round(Number($("v3-uq-n").value) || 16);
  const ns = sobol ? Math.max(12, n) : n;
  $("v3-uq").disabled = true;
  out.textContent = `running ${ns} 3-D solves…`;
  try {
    const d = await postJSON("uq", { n_samples: ns, params, method: sobol ? "sobol" : "correlation" });
    out.innerHTML = uqHtml(d);
  } catch (e) { out.textContent = e.message; } finally { $("v3-uq").disabled = false; }
}

// Inverse problem: fit one input so a model quantity matches a measured value
async function calibrate3d() {
  const out = $("v3-cal-out");
  const target = Number($("v3-cal-target").value);
  if (!Number.isFinite(target) || $("v3-cal-target").value === "") {
    out.textContent = "enter the measured target value"; return;
  }
  const param = $("v3-cal-param").value, qoi = $("v3-cal-qoi").value;
  $("v3-cal").disabled = true;
  out.textContent = "searching (one 3-D solve per step)…";
  try {
    const d = await postJSON("calibrate", { target, param, qoi,
      bounds: [Number($("v3-cal-lo").value), Number($("v3-cal-hi").value)] });
    const f = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4 ? v.toExponential(3) : v.toPrecision(4));
    const err = Math.abs(d.achieved - d.target) / (Math.abs(d.target) || 1);
    const [lo, hi] = [Number($("v3-cal-lo").value), Number($("v3-cal-hi").value)];
    const atBound = Math.min(Math.abs(d.value - lo), Math.abs(d.value - hi)) <= 1e-3 * (hi - lo);
    out.textContent = `${d.param} = ${f(d.value)} gives ${d.qoi} = ${f(d.achieved)} `
      + `(target ${f(d.target)}, ${(100 * err).toFixed(2)}% off) · ${d.n_eval} solves`;
    if (err > 0.01 && atBound) {
      out.textContent += ` — stopped at the search bound: the target is out of reach by `
        + `varying ${d.param} in [${lo}, ${hi}]. Widen the range or fit another input `
        + `(flow through a tree-limited organ barely responds to K).`;
    }
  } catch (e) { out.textContent = e.message; } finally { $("v3-cal").disabled = false; }
}
// sensible search bounds per fitted input
const CAL_BOUNDS = { permeability: [0.1, 10], p_in: [1, 50], mu: [0.2, 5] };
$("v3-cal-param").addEventListener("change", (e) => {
  const [lo, hi] = CAL_BOUNDS[e.target.value];
  $("v3-cal-lo").value = lo; $("v3-cal-hi").value = hi;
});
$("v3-cal-qoi").addEventListener("change", (e) => {   // the default target is an inflow
  const s = data && data.summary;
  $("v3-cal-target").value = e.target.value === "arterial_inflow" && s
    ? Number((1.2 * s.arterial_inflow).toPrecision(3)) : "";
});
$("v3-uq").addEventListener("click", () => uq3d());
$("v3-cal").addEventListener("click", () => calibrate3d());
for (const id of ["v3-mc-n", "v3-mc-beta"]) {
  $(id).addEventListener("change", () => loadProject().catch(() => {}));
}
$("v3-flow").addEventListener("click", () => flow3d().catch((e) => { $("v3-kin").textContent = e.message; }));
$("v3-grow").addEventListener("click", () => treeOp("grow"));
$("v3-prune").addEventListener("click", () => treeOp("prune"));
$("v3-adapt").addEventListener("click", () => treeOp("adapt"));
$("v3-k").addEventListener("change", () => loadProject().catch(() => {}));
$("v3-play").addEventListener("click", () => {
  anim.playing = !anim.playing; $("v3-play").textContent = anim.playing ? "⏸" : "▶";
});
$("v3-scrub").addEventListener("input", (e) => {
  anim.idx = Number(e.target.value); anim.playing = false; $("v3-play").textContent = "▶"; showFrame();
});
$("v3-run").addEventListener("click", () => run3d().catch((e) => { $("v3-run-info").textContent = e.message; }));
$("v3-field").addEventListener("change", () => loadProject().catch(() => {}));
$("v3-video-spin").addEventListener("change", syncVideo);
$("v3-export-fmt").addEventListener("change", (e) => { delete e.target.dataset.forced; });
$("v3-export").addEventListener("click", () => {
  window.location = `/projects/${PROJECT}/export?fmt=${$("v3-export-fmt").value}`;
});
$("v3-bundle").addEventListener("click", () => { window.location = `/projects/${PROJECT}/bundle`; });
$("v3-png").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = `${(data && data.tissue) || "perfusion"}-3d.png`;
  a.click();
});
$("v3-model").addEventListener("change", (e) => {
  $("v3-rhoc-row").hidden = e.target.value !== "direct";
  const stored = data && data.has_result ? (data.direct ? "direct" : "hybrid") : null;
  $("v3-run").textContent = stored === e.target.value ? "↻ Re-run 3-D simulation" : "▶ Run 3-D simulation";
});

function setData(d) {
  data = d;
  const [x0, x1, y0, y1, z0, z1] = data.domain;
  center = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
  span = Math.max(x1 - x0, y1 - y0, z1 - z0);
  const nPts = data.tracer ? data.tracer.xyz.length : data.pressure.voxels.length;
  $("info").textContent = data.direct
    ? `${data.segments.length} segments · ${data.direct.connections.length} terminal connections`
    : `${data.segments.length} segments · ${nPts} field points · grid ${data.shape.join("×")}`;
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

// Inverse of project(): a screen point at a given depth → world coordinates.
function unproject(sx, sy, depth) {
  const [rx, rz] = vp.toDomain(sx, sy);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const dy1 = depth * cp + rz * sp, dz = -depth * sp + rz * cp;     // undo the pitch
  const cy = Math.cos(view.yaw), sy0 = Math.sin(view.yaw);
  const dx = rx * cy + dy1 * sy0, dy = -rx * sy0 + dy1 * cy;          // undo the yaw
  return [center[0] + dx * span, center[1] + dy * span, center[2] + dz * span];
}

// Hand-editing the 3-D trees (3-D projects only; see edit3d.js)
const editor = PROJECT ? createEditor3d({
  canvas, vp, projectId: PROJECT, project, unproject,
  onInfo: (msg, err) => {
    $("v3-tree-info").textContent = msg;
    $("v3-tree-info").style.color = err ? "#b91c1c" : "";
  },
  onSaved: async () => {
    $("v3-field").value = "pressure"; $("v3-kin").textContent = "";
    await loadProject();
  },
}) : null;
// Lesions & hotspots (spheres) on a 3-D project (see regions3d.js)
const regions = PROJECT ? createRegions3d({
  projectId: PROJECT, project, unproject, scene: () => data,
  onInfo: (msg, err) => {
    $("v3-reg-info").textContent = msg;
    $("v3-reg-info").style.color = err ? "#b91c1c" : "";
  },
  onSaved: async () => { $("v3-field").value = "pressure"; await loadProject(); },
}) : null;
if (regions) {
  $("v3-add-lesion").addEventListener("click", () => regions.add("lesion"));
  $("v3-add-hotspot").addEventListener("click", () => regions.add("hotspot"));
}
if (editor) {
  window.viewer3d = { editor, regions, project };                   // hook for browser tests / tooling
  $("v3-edit").addEventListener("change", async (e) => {
    if (e.target.checked) { view.spin = false; $("spin").checked = false; }
    try { await editor.setActive(e.target.checked); }
    catch (err) { e.target.checked = false; $("v3-tree-info").textContent = err.message; }
  });
}

function render() {
  vp.begin();
  ctx.fillStyle = "#0b1020"; ctx.fillRect(0, 0, vp.W, vp.H);
  const zs = vp.scale / 500;                      // size factor that follows the zoom
  if (!data) return;

  if ($("show-field").checked && data.tracer) {          // flow video: one frame of the bolus
    const t = data.tracer, vals = t.frames[anim.idx], vmax = t.vmax || 1;
    const alpha = Number($("opacity").value) / 100;
    const pts = [];
    for (let i = 0; i < t.xyz.length; i++) {
      const c = vals[i] / vmax;
      if (c < 0.02) continue;                          // empty tissue stays transparent
      const [sx, sy, depth] = project(t.xyz[i][0], t.xyz[i][1], t.xyz[i][2]);
      pts.push({ sx, sy, depth, t: Math.min(1, c) });
    }
    pts.sort((a, b) => a.depth - b.depth);
    for (const p of pts) {
      ctx.globalAlpha = alpha * (0.35 + 0.65 * p.t);
      ctx.fillStyle = viridisCss(p.t);
      const r = Math.max(1.2, 2.6 * zs);
      ctx.fillRect(p.sx - r, p.sy - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
  } else if ($("show-field").checked) {
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

  if (!data.pressure.voxels.length && data.tissue_points && !data.tracer) {   // unsolved 3-D project
    ctx.fillStyle = data.direct ? "rgba(148,163,184,.16)" : "rgba(52,211,153,.35)";   // grey under green links
    for (const v of data.tissue_points) {
      const [sx, sy] = project(v[0], v[1], v[2]);
      ctx.fillRect(sx - 1, sy - 1, 2, 2);
    }
  }

  if (regions && data.dim === 3) regions.draw(ctx);
  if (editor && editor.active) {
    editor.draw(ctx, zs);
  } else if (data.direct && $("show-trees").checked) {
    drawDirect(zs);
  } else if ($("show-trees").checked) {
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

// A direct-connection result: the trees and their terminal links (the capillary
// bed, green) weighted by flow — or, with a network bolus, each element lit by its
// concentration in the current frame. Lines are drawn far-to-near.
function drawDirect(zs) {
  const d = data.direct, nt = data.net_tracer;
  const items = [];
  if (nt) {
    const vals = nt.frames[anim.idx], vmax = nt.vmax || 1;
    nt.lines.forEach((l, i) => items.push({ l, kind: nt.kind[i], t: Math.min(1, vals[i] / vmax), bolus: true }));
  } else {
    const fmax = d.flow_max || 1;
    for (const [rows, kind] of [[d.artery, 0], [d.vein, 1], [d.connections, 2]]) {
      for (const r of rows) items.push({ l: r, kind, t: Math.min(1, r[6] / fmax) });
    }
  }
  const pts = items.map((it) => {
    const a = project(it.l[0], it.l[1], it.l[2]), b = project(it.l[3], it.l[4], it.l[5]);
    return { ...it, a, b, depth: (a[2] + b[2]) / 2 };
  }).sort((p, q) => p.depth - q.depth);
  ctx.lineCap = "round";
  const color = ["#ff6b6b", "#5b8dff", "#10b981"];
  for (const p of pts) {
    if (p.bolus) {            // dim network, bright bolus
      ctx.globalAlpha = 0.25 + 0.75 * p.t;
      ctx.strokeStyle = p.t > 0.02 ? viridisCss(p.t) : "#334155";
      ctx.lineWidth = Math.max(1, (p.kind === 2 ? 1.2 : 2.4) * zs * (0.6 + p.t));
    } else if (p.kind === 2) {
      ctx.globalAlpha = 0.25 + 0.5 * Math.sqrt(p.t);
      ctx.strokeStyle = color[2]; ctx.lineWidth = Math.max(0.7, 0.9 * zs);
    } else {
      ctx.globalAlpha = 0.4 + 0.6 * p.t;
      ctx.strokeStyle = color[p.kind]; ctx.lineWidth = Math.max(1, 5 * Math.sqrt(p.t) * zs);
    }
    ctx.beginPath(); ctx.moveTo(p.a[0], p.a[1]); ctx.lineTo(p.b[0], p.b[1]); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function loop(now) {
  if (view.spin && data) view.yaw += 0.006;
  const tr = tracerOf(data);
  if (tr && anim.playing && now - anim.last > 110) {   // ~9 frames/s
    anim.last = now;
    anim.idx = (anim.idx + 1) % tr.frames.length;
    showFrame();
  }
  render();
  requestAnimationFrame(loop);
}

// ── interaction ────────────────────────────────────────────────────────────
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  view.spin = false; $("spin").checked = false;
  if (editor && editor.pointerDown(e)) return;        // picked / dragging a node
  if (regions && data && data.dim === 3 && regions.pointerDown(canvas, e)) return;
  dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("pointerup", () => {
  dragging = false;
  if (editor) editor.pointerUp();
  if (regions) regions.pointerUp();
});
window.addEventListener("pointermove", (e) => {
  if (editor && editor.pointerMove(e)) return;
  if (regions && regions.pointerMove(canvas, e)) return;
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
requestAnimationFrame(loop);
