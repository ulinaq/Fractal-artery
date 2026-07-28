// perfusion — 2-D workspace front-end (no build step, ES modules).
// Talks to the FastAPI back-end that serves this file.

import {
  canvas, ctx, W, H, DOMAIN, GRID_L, toCanvas, toDomain, api, state, status,
  $, numOf, vtype, getCss, hasEditable, editKinds,
} from "./core.js";
import { redraw } from "./render.js";
import {
  deleteSelectedNode, editPointerDown, ensureNetworkForEdit, loadEditable, markEditDirty,
  moveNode, onEditRadius, revertEdits, saveEdits, syncEditPanel, toggleAddMode,
} from "./edit.js";
import {
  adapt, computeFlow, fitKinetics, grow, loadField, loadKineticMap, loadMulticompartment,
  loadNetwork, loadOxygen, loadResultView, pauseFlow, run, runTransient, stopFlow,
  stopTimer, syncFlowUI, toggleFlowPlay,
} from "./run.js";

// ── undo / redo (workspace geometry history) ───────────────────────────────────
// Snapshots the editable geometry after each action — vessel network, hand-drawn
// strokes, tissue regions and inlet/outlet points — so any edit can be reverted.
const history = { stack: [], idx: -1 };

function geomSnapshot() {
  return JSON.stringify({
    vessels: state.vessels, regions: state.regions, lesions: state.lesions,
    inlets: state.inlets, outlets: state.outlets,
    editable: state.editable, editSource: state.editSource,
  });
}

export function recordHistory() {
  const snap = geomSnapshot();
  if (history.idx >= 0 && history.stack[history.idx] === snap) return;   // no real change
  history.stack = history.stack.slice(0, history.idx + 1);               // drop redo tail
  history.stack.push(snap);
  if (history.stack.length > 100) history.stack.shift();
  history.idx = history.stack.length - 1;
  updateHistoryButtons();
}

function resetHistory() {
  history.stack = [geomSnapshot()];
  history.idx = 0;
  updateHistoryButtons();
}

function updateHistoryButtons() {
  $("btn-hist-undo").disabled = history.idx <= 0 || state.busy;
  $("btn-hist-redo").disabled = history.idx >= history.stack.length - 1 || state.busy;
}

function applySnapshot(snap) {
  const s = JSON.parse(snap);
  state.vessels = s.vessels || { arteries: [], veins: [] };
  state.regions = s.regions || [];
  state.lesions = s.lesions || [];
  state.inlets = s.inlets || []; state.outlets = s.outlets || [];
  state.editable = s.editable; state.editSource = s.editSource;
  state.editSel = null; state.editAdd = false; state.editDrag = false;
  $("btn-edit-add").classList.remove("active");
  if (state.editSource) markEditDirty();     // differs from the saved geometry -> Save to persist
  syncEditPanel(); refreshUI(); redraw();
}

function historyUndo() {
  if (history.idx <= 0 || state.busy) return;
  applySnapshot(history.stack[--history.idx]);
  updateHistoryButtons();
  status("undo", "ok");
}

function historyRedo() {
  if (history.idx >= history.stack.length - 1 || state.busy) return;
  applySnapshot(history.stack[++history.idx]);
  updateHistoryButtons();
  status("redo", "ok");
}

// Actions that require a project (disabled until one exists, or while busy).
const NEEDS_PROJECT = [
  "btn-save-file",
  "btn-endseg", "btn-undo", "btn-clear", "btn-save-vessels", "img-file",
  "btn-endregion", "btn-undo-region", "btn-clear-tissue", "btn-save-mask",
  "btn-clear-io", "btn-save-io", "btn-edit-save", "btn-edit-revert",
  "btn-grow", "btn-adapt", "btn-run", "btn-flow",
];

export function refreshUI() {
  const ready = !!state.project && !state.busy;
  NEEDS_PROJECT.forEach((id) => { $(id).disabled = !ready; });
  $("btn-new").disabled = state.busy;
  $("btn-frog").disabled = state.busy;
  $("btn-load-file").disabled = state.busy;
  const canExport = !!(state.project && state.hasResult) && !state.busy
    && state.resultModel !== "direct";        // direct has no grid field to export
  $("btn-export").disabled = !canExport;
  $("btn-bundle").disabled = !canExport;
  updateHistoryButtons();
}

function setBusy(on) { state.busy = on; refreshUI(); }

// Wrap a long-running action: lock the UI, surface errors, always unlock.
function guarded(fn) {
  return async () => {
    if (state.busy) return;
    setBusy(true);
    try { await fn(); }
    catch (e) { status(e.message, "err"); }
    finally { setBusy(false); }
  };
}

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
  state.stroke = []; state.regions = []; state.regionStroke = [];
  state.lesions = []; state.lesionStroke = [];
  state.inlets = []; state.outlets = [];
  state.network = null; state.fieldData = null;
  stopFlow(); state.tracer = null; state.hasResult = false;
  showProjectLink(p.id);
  $("grow-info").textContent = ""; $("run-info").textContent = "";
  setMode("vessels");
  refreshUI();
  status("project ready", "ok");
  resetHistory();
  redraw();
}

function showProjectLink(id) {
  const url = `${location.origin}/?project=${id}`;
  $("proj-id").innerHTML = `project <a href="${url}" title="shareable link">${id}</a>`;
  $("link-3d").href = `/viewer3d.html?project=${id}`;    // 3-D view keeps this project
}

async function loadFrog() {
  status("segmenting the frog example…");
  let p;
  try {
    p = await api("POST", "/projects/frog");
  } catch (e) {
    status(/50[01]|image/.test(e.message)
      ? "Frog needs the 'image' extra — restart: uv run --extra api --extra image perfusion-api"
      : e.message, "err");
    return;
  }
  await openProject(p.id);
  $("use-vessels").checked = true;
  status("frog example loaded — press Run to solve it", "ok");
}

// ── save / load a project as a portable .json file ─────────────────────────────
async function saveToFile() {
  requireProject();
  // the endpoint sets Content-Disposition: attachment, so this downloads (no nav)
  window.location = `/projects/${state.project}/document`;
  status("project saved to file", "ok");
}

async function loadFromFile() {
  const file = $("file-load").files[0];
  if (!file) return;
  status("loading project file…");
  let doc;
  try { doc = JSON.parse(await file.text()); }
  catch { status("not a valid project .json file", "err"); return; }
  const info = await api("POST", "/projects/import", doc);
  await openProject(info.id);
  $("file-load").value = "";                 // allow re-loading the same file
  status(`loaded “${info.name}” from file`, "ok");
}

export function requireProject() {
  if (!state.project) { status("create a project first", "err"); throw new Error("no project"); }
  return state.project;
}

// ── vessel drawing ────────────────────────────────────────────────────────────
const canvasXY = (e) => {
  const rect = canvas.getBoundingClientRect();
  return [(e.clientX - rect.left) * (W / rect.width),
          (e.clientY - rect.top) * (H / rect.height)];
};

canvas.addEventListener("pointerdown", (e) => {
  if (state.mode === "result") return;
  const [px, py] = canvasXY(e);
  if (state.mode === "edit") { editPointerDown(px, py); return; }
  const pt = toDomain(px, py);
  if (state.mode === "vessels") state.stroke.push(pt);
  else if (state.mode === "tissue") (drawingLesion() ? state.lesionStroke : state.regionStroke).push(pt);
  else if (state.mode === "io") markIO(pt);
  redraw();
});

canvas.addEventListener("pointermove", (e) => {
  if (state.mode !== "edit" || !state.editDrag || !state.editSel) return;
  const [px, py] = canvasXY(e);
  const [x, y] = toDomain(px, py);
  moveNode(state.editSel.kind, state.editSel.node, x, y);
  redraw();
});

const endEditDrag = () => {
  if (state.editDrag) { state.editDrag = false; markEditDirty(); recordHistory(); }
};
canvas.addEventListener("pointerup", endEditDrag);
canvas.addEventListener("pointerleave", endEditDrag);

// place/remove an inlet or outlet near the click (own toggle in the I/O card)
function markIO(pt) {
  const isInlet = document.querySelector("input[name=iotype]:checked").value === "inlet";
  const list = isInlet ? state.inlets : state.outlets;
  const tol = 4;   // domain units: click a marker again to remove it
  const hit = list.findIndex(([x, y]) => Math.hypot(x - pt[0], y - pt[1]) < tol);
  if (hit >= 0) list.splice(hit, 1);
  else list.push([pt[0], pt[1]]);
  recordHistory();
}

function endStroke() {
  if (state.stroke.length >= 2) {
    const bucket = vtype() === "artery" ? state.vessels.arteries : state.vessels.veins;
    bucket.push({ points: state.stroke.map(([x, y]) => [x, y]), radius: numOf("draw-radius") });
    recordHistory();
  }
  state.stroke = [];
  redraw();
}
function undo() {
  if (state.stroke.length) { state.stroke.pop(); }
  else {
    const bucket = vtype() === "artery" ? state.vessels.arteries : state.vessels.veins;
    bucket.pop();
    recordHistory();
  }
  redraw();
}
function clearVessels() { state.vessels = { arteries: [], veins: [] }; state.stroke = []; recordHistory(); redraw(); }

export async function saveVessels() {
  requireProject();
  await api("PUT", `/projects/${state.project}/vessels`, state.vessels);
  $("use-vessels").checked = true;
  status(`saved ${state.vessels.arteries.length} arteries, ${state.vessels.veins.length} veins`, "ok");
}

// ── tissue mask + permeability-lesion drawing ───────────────────────────────────
function drawingLesion() { return $("tissue-target").value === "lesion"; }

function endRegion() {
  if (drawingLesion()) {
    if (state.lesionStroke.length >= 3) {
      state.lesions.push({ polygon: state.lesionStroke.map(([x, y]) => [x, y]),
                           permeability: Math.max(1e-3, numOf("lesion-k")) });
      recordHistory();
    }
    state.lesionStroke = [];
  } else {
    if (state.regionStroke.length >= 3) {
      state.regions.push(state.regionStroke.map(([x, y]) => [x, y]));
      recordHistory();
    }
    state.regionStroke = [];
  }
  redraw();
}
function undoRegion() {
  if (drawingLesion()) {
    if (state.lesionStroke.length) state.lesionStroke.pop();
    else { state.lesions.pop(); recordHistory(); }
  } else if (state.regionStroke.length) {
    state.regionStroke.pop();
  } else { state.regions.pop(); recordHistory(); }
  redraw();
}
function clearTissue() {
  if (drawingLesion()) { state.lesions = []; state.lesionStroke = []; }
  else { state.regions = []; state.regionStroke = []; }
  recordHistory(); redraw();
}

export async function saveMask() {
  requireProject();
  if (state.regionStroke.length >= 3 || state.lesionStroke.length >= 3) endRegion();
  await api("PUT", `/projects/${state.project}/mask`, { regions: state.regions });
  await api("PUT", `/projects/${state.project}/lesions`, { lesions: state.lesions });
  const bits = [];
  bits.push(state.regions.length ? `${state.regions.length} region(s)` : "full domain");
  if (state.lesions.length) bits.push(`${state.lesions.length} lesion(s)`);
  status(`saved tissue — ${bits.join(", ")}`, "ok");
}

async function loadMask() {
  try { state.regions = (await api("GET", `/projects/${state.project}/mask`)).regions || []; }
  catch { state.regions = []; }
  try { state.lesions = (await api("GET", `/projects/${state.project}/lesions`)).lesions || []; }
  catch { state.lesions = []; }
}

export async function loadVessels() {
  try {
    const v = await api("GET", `/projects/${state.project}/vessels`);
    state.vessels = { arteries: v.arteries || [], veins: v.veins || [] };
  } catch { state.vessels = { arteries: [], veins: [] }; }
}

// ── inlets / outlets ──────────────────────────────────────────────────────────
function clearIO() { state.inlets = []; state.outlets = []; recordHistory(); redraw(); }

export async function saveIO() {
  requireProject();
  await api("PUT", `/projects/${state.project}/io`,
            { inlets: state.inlets, outlets: state.outlets });
  $("io-info").textContent = `${state.inlets.length} inlet(s), ${state.outlets.length} outlet(s)`;
  status("inlets/outlets saved", "ok");
}

async function loadIO() {
  try {
    const io = await api("GET", `/projects/${state.project}/io`);
    state.inlets = io.inlets || [];
    state.outlets = io.outlets || [];
  } catch { state.inlets = []; state.outlets = []; }
}

async function uploadImage(file) {
  requireProject();
  const kind = vtype();                    // artery/vein selector governs the target
  status(`segmenting ${kind} image…`);
  const r = await fetch(`/projects/${state.project}/image?kind=${kind}`, { method: "POST", body: file });
  if (!r.ok) { status(`segment failed (${r.status}): ${(await r.text()).slice(0, 140)}`, "err"); return; }
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  const n = state.network ? state.network.segments.length : 0;
  $("img-info").textContent = `segmented → ${n} total segments`;
  status(`image segmented → ${kind} tree`, "ok");
  redraw();
}

// ── modes ────────────────────────────────────────────────────────────────────
export function setMode(mode) {
  state.mode = mode;
  $("mode-vessels").classList.toggle("active", mode === "vessels");
  $("mode-tissue").classList.toggle("active", mode === "tissue");
  $("mode-io").classList.toggle("active", mode === "io");
  $("mode-edit").classList.toggle("active", mode === "edit");
  $("mode-result").classList.toggle("active", mode === "result");
  const fieldResult = mode === "result" && state.resultModel !== "direct";
  $("field-picker").classList.toggle("show", fieldResult);   // direct has no field to pick
  $("flowbar").classList.toggle("show", mode === "result");  // tracer: field (hybrid) or network (direct)
  $("colorbar").hidden = !fieldResult;                       // network animation unhides it on demand
  // show only the tool card(s) for the active mode (Project/Parameters/Run stay)
  document.querySelectorAll(".card[data-mode]").forEach((c) => {
    c.style.display = c.dataset.mode === mode ? "" : "none";
  });
  if (mode !== "result") stopTimer();
  state.editAdd = false; $("btn-edit-add").classList.remove("active");
  canvas.style.cursor = mode === "result" ? "default" : mode === "edit" ? "pointer" : "crosshair";
  if (mode === "edit") {
    syncEditPanel();
    // fold any hand-drawn strokes into the vessel network so Edit is graph-based
    if (!state.editSource) ensureNetworkForEdit();
  }
  redraw();
}

// ── wiring ────────────────────────────────────────────────────────────────────
$("btn-new").addEventListener("click", guarded(newProject));
$("btn-frog").addEventListener("click", guarded(loadFrog));
$("btn-save-file").addEventListener("click", guarded(saveToFile));
$("btn-load-file").addEventListener("click", () => $("file-load").click());
$("file-load").addEventListener("change", guarded(loadFromFile));
$("edit-radius").addEventListener("input", onEditRadius);
$("edit-radius").addEventListener("change", recordHistory);   // one history entry per resize
$("btn-hist-undo").addEventListener("click", historyUndo);
$("btn-hist-redo").addEventListener("click", historyRedo);

document.addEventListener("keydown", (e) => {
  if (!(e.metaKey || e.ctrlKey)) return;
  if (e.key.toLowerCase() !== "z" && e.key.toLowerCase() !== "y") return;
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;   // let text fields keep their own undo
  e.preventDefault();
  const redo = (e.key.toLowerCase() === "z" && e.shiftKey) || e.key.toLowerCase() === "y";
  redo ? historyRedo() : historyUndo();
});

// ── context-aware keyboard shortcuts (no modifiers, outside form fields) ────────
// Speeds up the repeated actions: Del deletes the selected node, Enter ends a
// stroke/region, A adds a branch / picks artery, Esc cancels, R runs, G grows,
// 1–5 switch tabs. Clicking the real buttons reuses their enabled/guarded logic.
const clickIfEnabled = (id) => { const b = $(id); if (b && !b.disabled) b.click(); };

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;                // combos handled above / by the browser
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;   // don't hijack typing in fields
  const m = state.mode, k = e.key.toLowerCase();
  let handled = true;
  if (e.key === "Delete" || e.key === "Backspace") {
    if (m === "edit") clickIfEnabled("btn-edit-del");            // delete selected node
    else if (m === "vessels") clickIfEnabled("btn-undo");        // remove last stroke
    else if (m === "tissue") clickIfEnabled("btn-undo-region");
    else handled = false;
  } else if (e.key === "Enter") {
    if (m === "vessels") clickIfEnabled("btn-endseg");           // end the current stroke
    else if (m === "tissue") clickIfEnabled("btn-endregion");    // close the region
    else handled = false;
  } else if (e.key === "Escape") {
    if (m === "edit") { if (state.editAdd) toggleAddMode(); else { state.editSel = null; syncEditPanel(); redraw(); } }
    else if (m === "vessels" && state.stroke.length) { state.stroke = []; redraw(); }
    else if (m === "tissue" && (state.regionStroke.length || state.lesionStroke.length)) {
      state.regionStroke = []; state.lesionStroke = []; redraw();
    }
    else handled = false;
  } else if (k === "a") {
    if (m === "edit") clickIfEnabled("btn-edit-add");            // toggle add-branch
    else if (m === "vessels") document.querySelector('input[name=vtype][value=artery]').checked = true;
    else handled = false;
  } else if (k === "v" && m === "vessels") {
    document.querySelector('input[name=vtype][value=vein]').checked = true;
  } else if (k === "g" && m === "vessels") {
    clickIfEnabled("btn-grow");
  } else if (k === "r") {
    clickIfEnabled("btn-run");                                   // run simulation
  } else if (e.key >= "1" && e.key <= "5") {
    setMode(["vessels", "tissue", "io", "edit", "result"][+e.key - 1]);
  } else {
    handled = false;
  }
  if (handled) e.preventDefault();
});
$("btn-edit-add").addEventListener("click", toggleAddMode);
$("btn-edit-del").addEventListener("click", deleteSelectedNode);
$("btn-edit-save").addEventListener("click", guarded(saveEdits));
$("btn-edit-revert").addEventListener("click", guarded(revertEdits));
$("btn-endseg").addEventListener("click", endStroke);
$("btn-undo").addEventListener("click", undo);
$("btn-clear").addEventListener("click", clearVessels);
$("btn-save-vessels").addEventListener("click", () => saveVessels().catch((e) => status(e.message, "err")));
$("img-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) uploadImage(f).catch((err) => status(err.message, "err"));
});
$("btn-endregion").addEventListener("click", endRegion);
$("btn-undo-region").addEventListener("click", undoRegion);
$("btn-clear-tissue").addEventListener("click", clearTissue);
$("tissue-target").addEventListener("change", () => {
  $("lesion-k-row").style.display = drawingLesion() ? "" : "none";
});
$("btn-save-mask").addEventListener("click", () => saveMask().catch((e) => status(e.message, "err")));
$("btn-grow").addEventListener("click", guarded(grow));
$("btn-adapt").addEventListener("click", guarded(adapt));
$("btn-run").addEventListener("click", guarded(run));
$("nonlinear").addEventListener("change", (e) => {   // max-iter only matters for the nonlinear solve
  $("maxiter-row").style.display = e.target.checked ? "" : "none";
});
$("elasticity").addEventListener("change", (e) => {  // compliance only matters with elasticity on
  $("compliance-row").style.display = e.target.checked ? "" : "none";
});
$("run-model").addEventListener("change", (e) => {   // show the settings each model needs
  const direct = e.target.value === "direct";
  $("rhoc-row").style.display = direct ? "" : "none";
  $("nonlinear-row").style.display = direct ? "none" : "";
  $("phasesep-row").style.display = direct ? "none" : "";
  $("elastic-row").style.display = direct ? "none" : "";
  $("compliance-row").style.display = (!direct && $("elasticity").checked) ? "" : "none";
  $("maxiter-row").style.display = (!direct && $("nonlinear").checked) ? "" : "none";
});
$("btn-export").addEventListener("click", () => {
  window.location = `/projects/${state.project}/export?fmt=${$("export-fmt").value}`;
});
$("btn-bundle").addEventListener("click", () => {
  window.location = `/projects/${state.project}/bundle`;
});
$("btn-png").addEventListener("click", () => {
  // client-side snapshot of the current view, composited onto a white background
  const out = document.createElement("canvas");
  out.width = canvas.width; out.height = canvas.height;
  const octx = out.getContext("2d");
  octx.fillStyle = "#fff"; octx.fillRect(0, 0, out.width, out.height);
  octx.drawImage(canvas, 0, 0);
  out.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `perfusion-${state.project || "view"}-${state.mode}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
});
$("btn-clear-io").addEventListener("click", clearIO);
$("btn-save-io").addEventListener("click", () => saveIO().catch((e) => status(e.message, "err")));
$("mode-vessels").addEventListener("click", () => setMode("vessels"));
$("mode-tissue").addEventListener("click", () => setMode("tissue"));
$("mode-io").addEventListener("click", () => setMode("io"));
$("mode-edit").addEventListener("click", () => setMode("edit"));
$("mode-result").addEventListener("click", () => setMode("result"));
canvas.addEventListener("dblclick", () => (state.mode === "tissue" ? endRegion() : endStroke()));
document.querySelectorAll(".field").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".field").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  stopFlow();                              // switch from animation back to a static field
  if (state.project) loadField(b.dataset.field).catch((e) => status(e.message, "err"));
}));

$("btn-flow").addEventListener("click", guarded(computeFlow));
$("btn-tkm").addEventListener("click", guarded(fitKinetics));
$("btn-transient").addEventListener("click", guarded(runTransient));
const mapMethod = () => ($("map-deconv").checked ? "deconv" : "semi");
$("btn-map-fp").addEventListener("click", guarded(() => loadKineticMap("Fp", mapMethod())));
$("btn-map-vp").addEventListener("click",
  guarded(() => loadKineticMap(mapMethod() === "deconv" ? "CBV" : "vp", mapMethod())));
$("btn-map-mtt").addEventListener("click", guarded(() => loadKineticMap("MTT", "deconv")));
$("btn-o2-sat").addEventListener("click", guarded(() => loadOxygen("saturation")));
$("btn-o2-cons").addEventListener("click", guarded(() => loadOxygen("consumption")));
$("btn-mc-perf").addEventListener("click", guarded(() => loadMulticompartment("perfusion")));
$("btn-mc-comp0").addEventListener("click", guarded(() => loadMulticompartment("arteriole")));
$("btn-mc-compN").addEventListener("click", guarded(() => loadMulticompartment("venule")));
$("tkm-model").addEventListener("change", () => { if (state.tracer) fitKinetics().catch(() => {}); });
$("flow-play").addEventListener("click", toggleFlowPlay);
$("flow-scrub").addEventListener("input", (e) => {
  pauseFlow(); state.flow.idx = Number(e.target.value); syncFlowUI(); redraw();
});

// Reflect a project's stored parameters/grid in the input fields.
function syncParamInputs(info) {
  const p = info.params, g = info.grid;
  if (p) {
    if (p.p_in != null) $("p_in").value = p.p_in;
    if (p.p_out != null) $("p_out").value = p.p_out;
    if (p.r_min_frac != null) $("r_min_frac").value = p.r_min_frac;
    if (p.seed != null) $("seed").value = p.seed;
  }
  if (g) {
    if (g.nx != null) $("nx").value = g.nx;
    if (g.ny != null) $("ny").value = g.ny;
  }
}

// Open an existing project by id (shareable URL: ?project=<id>).
async function openProject(id) {
  const info = await api("GET", `/projects/${id}`);
  state.project = info.id;
  $("proj-name").value = info.name;
  syncParamInputs(info);                     // reflect the project's stored settings
  showProjectLink(info.id);
  stopFlow(); state.tracer = null;
  state.hasResult = !!info.has_result;
  await loadVessels();                       // restore hand-drawn strokes first
  await Promise.all([loadNetwork(), loadMask(), loadIO(), loadEditable()]);
  if (info.has_result) {
    await loadResultView();                  // field or direct-connection view
    setMode("result");
  } else { setMode("vessels"); }
  refreshUI();
  resetHistory();
  status(`opened ${info.id}`, "ok");
}

// ── boot ──────────────────────────────────────────────────────────────────────
(async function boot() {
  refreshUI();                               // actions disabled until a project exists
  setMode("vessels");                        // tidy: show only the active mode's tools
  try { const h = await api("GET", "/health"); status(`connected · v${h.version}`, "ok"); }
  catch { status("back-end unreachable", "err"); redraw(); return; }
  const shared = new URLSearchParams(location.search).get("project");
  if (shared) { try { await openProject(shared); return; } catch (e) { status(e.message, "err"); } }
  redraw();
})();
