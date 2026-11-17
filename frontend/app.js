// perfusion — 2-D workspace front-end (no build step, ES modules).
// Talks to the FastAPI back-end that serves this file.
//
// Three tabs: Vessels (edit the artery/vein *networks* node by node — edit.js),
// Tissue (edit the organ/lesion/hotspot polygons vertex by vertex — tissue.js)
// and Result. This module owns project lifecycle, history, modes and wiring.

import { canvas, W, H, DOMAIN, GRID_L, api, state, status, $, numOf } from "./core.js";
import { redraw } from "./render.js";
import {
  adoptLegacyIO, clearEditDirty, clearKind, deleteSelectedNode, deselect, editPointerDown,
  editDoubleClick, editPointerMove, editPointerUp, flushEdits, insertNodeOnSelected,
  loadEditable, markEditDirty, newKind, onRadiusInput, onRadiusSlider, revertEdits, saveEdits, syncEditPanel, toggleInlet, vectorizeLegacy,
} from "./edit.js";
import {
  cancelOrDeselect, clearTarget, clearTissueDirty, closePolygon, deleteSelection, flushTissue,
  loadMask, markDirty as markTissueDirty, onTissueValue, saveMask, syncTargetRows,
  syncTissuePanel, tissuePointerDown, tissuePointerMove, tissuePointerUp,
} from "./tissue.js";
import {
  adapt, computeFlow, fitKinetics, grow, loadField, loadKineticMap, loadMorphometry,
  loadMulticompartment, loadNetwork, loadOxygen, loadResultView, pauseFlow, prune, run,
  runTransient, runUQ, stopFlow, stopTimer, syncFlowUI, toggleFlowPlay,
} from "./run.js";

// ── undo / redo (workspace geometry history) ───────────────────────────────────
// Snapshots the editable geometry after each action — vessel networks (incl.
// inlet/outlet nodes) and tissue polygons — so any edit can be reverted.
const history = { stack: [], idx: -1 };

function geomSnapshot() {
  return JSON.stringify({
    regions: state.regions, lesions: state.lesions, hotspots: state.hotspots,
    editable: state.editable,
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
  const prev = JSON.parse(geomSnapshot()), s = JSON.parse(snap);
  state.regions = s.regions || [];
  state.lesions = s.lesions || [];
  state.hotspots = s.hotspots || [];
  state.editable = s.editable || { artery: null, vein: null };
  state.editSel = null; state.tissueSel = null; state.tissueStroke = [];
  // whatever differs from the current geometry now differs from the saved copy
  if (JSON.stringify(prev.editable) !== JSON.stringify(s.editable)) markEditDirty();
  if (JSON.stringify([prev.regions, prev.lesions, prev.hotspots])
      !== JSON.stringify([s.regions, s.lesions, s.hotspots])) markTissueDirty();
  syncEditPanel(); syncTissuePanel(); refreshUI(); redraw();
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
  "btn-save-file", "btn-save-vessels", "btn-edit-revert", "btn-clear-kind", "img-file",
  "btn-save-mask", "btn-tissue-revert", "btn-tissue-close", "btn-clear-tissue",
  "btn-grow", "btn-prune", "btn-adapt", "btn-run", "btn-flow",
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
  syncEditPanel(); syncTissuePanel();
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
function resetGeometry() {
  state.vessels = { arteries: [], veins: [] };
  state.regions = []; state.lesions = []; state.hotspots = [];
  state.tissueStroke = []; state.tissueSel = null;
  state.inlets = []; state.outlets = []; state.legacyIO = false;
  state.editable = { artery: null, vein: null }; state.editSel = null;
  state.network = null; state.fieldData = null; state.direct = null;
  $("summary").innerHTML = ""; $("run-info").textContent = "";   // previous project's result
  clearEditDirty(); clearTissueDirty();
}

async function newProject() {
  const body = {
    name: $("proj-name").value || "workspace",
    params: { p_in: numOf("p_in"), p_out: numOf("p_out"),
              r_min_frac: numOf("r_min_frac"), seed: numOf("seed") },
    grid: { nx: numOf("nx"), ny: numOf("ny"), ...GRID_L,
            domain_limit: [DOMAIN.xmin, DOMAIN.xmax, DOMAIN.ymin, DOMAIN.ymax] },
  };
  const ky = numOf("perm_y");
  if (ky && ky !== 1) body.grid.permeability_y = ky;   // anisotropic capillary bed (Kx = 1)
  const p = await api("POST", "/projects", body);
  state.project = p.id;
  resetGeometry();
  stopFlow(); state.tracer = null; state.hasResult = false;
  showProjectLink(p.id);
  $("grow-info").textContent = ""; $("run-info").textContent = "";
  setMode("vessels");
  refreshUI();
  status("project ready — click the canvas to draw a vessel", "ok");
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
  status("frog example loaded — press Run to solve it", "ok");
}

// Persist any unsaved vessel/tissue edits (before run, grow, export…).
export async function flushAll() {
  await flushEdits();
  await flushTissue();
}

// ── save / load a project as a portable .json file ─────────────────────────────
async function saveToFile() {
  requireProject();
  await flushAll();
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

// ── legacy geometry loaders (pre-network projects) ─────────────────────────────
export async function loadVessels() {
  try {
    const v = await api("GET", `/projects/${state.project}/vessels`);
    state.vessels = { arteries: v.arteries || [], veins: v.veins || [] };
  } catch { state.vessels = { arteries: [], veins: [] }; }
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
  await flushEdits();
  const kind = newKind();                  // artery/vein selector governs the target
  status(`segmenting ${kind} image…`);
  const r = await fetch(`/projects/${state.project}/image?kind=${kind}`, { method: "POST", body: file });
  if (!r.ok) { status(`segment failed (${r.status}): ${(await r.text()).slice(0, 140)}`, "err"); return; }
  stopFlow(); state.tracer = null; state.fieldData = null; state.hasResult = false;
  await Promise.all([loadNetwork(), loadEditable()]);
  const n = state.network ? state.network.segments.length : 0;
  $("img-info").textContent = `segmented → ${n} total segments`;
  status(`image segmented → ${kind} tree`, "ok");
  recordHistory(); refreshUI(); redraw();
}

// ── canvas pointer routing ────────────────────────────────────────────────────
const canvasXY = (e) => {
  const rect = canvas.getBoundingClientRect();
  return [(e.clientX - rect.left) * (W / rect.width),
          (e.clientY - rect.top) * (H / rect.height)];
};

canvas.addEventListener("pointerdown", (e) => {
  if (state.mode === "result" || state.busy || e.button !== 0) return;
  canvas.setPointerCapture(e.pointerId);         // keep dragging even off-canvas
  const [px, py] = canvasXY(e);
  if (state.mode === "vessels") editPointerDown(px, py, e);
  else if (state.mode === "tissue") tissuePointerDown(px, py);
});

canvas.addEventListener("pointermove", (e) => {
  if (state.mode === "result") return;
  const [px, py] = canvasXY(e);
  if (state.mode === "vessels") editPointerMove(px, py);
  else if (state.mode === "tissue") tissuePointerMove(px, py);
});

canvas.addEventListener("pointerup", (e) => {
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  if (state.mode === "vessels") editPointerUp();
  else if (state.mode === "tissue") tissuePointerUp();
});

canvas.addEventListener("pointerleave", () => {
  if (state.hover) { state.hover = null; if (state.mode !== "result") redraw(); }
});

canvas.addEventListener("dblclick", (e) => {
  if (state.mode === "tissue" && state.tissueStroke.length) closePolygon();
  else if (state.mode === "vessels" && !state.busy) editDoubleClick(...canvasXY(e));
});

// ── modes ────────────────────────────────────────────────────────────────────
const MODES = ["vessels", "tissue", "result"];

export function setMode(mode) {
  if (state.mode === "vessels" && mode !== "vessels" && state.editSel) deselect();
  if (state.mode === "tissue" && mode !== "tissue") { state.tissueStroke = []; state.tissueSel = null; }
  state.mode = mode;
  state.hover = null;
  MODES.forEach((m) => $(`mode-${m}`).classList.toggle("active", mode === m));
  const fieldResult = mode === "result" && state.resultModel !== "direct";
  $("field-picker").classList.toggle("show", fieldResult);   // direct has no field to pick
  $("flowbar").classList.toggle("show", mode === "result");  // tracer: field (hybrid) or network (direct)
  $("colorbar").hidden = !fieldResult;                       // network animation unhides it on demand
  // show only the tool card(s) for the active mode (Project/Parameters/Run stay)
  document.querySelectorAll(".card[data-mode]").forEach((c) => {
    c.style.display = c.dataset.mode === mode ? "" : "none";
  });
  if (mode !== "result") stopTimer();
  canvas.style.cursor = mode === "result" ? "default" : "crosshair";
  syncEditPanel(); syncTissuePanel();
  redraw();
}

// ── keyboard ─────────────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (!(e.metaKey || e.ctrlKey)) return;
  if (e.key.toLowerCase() !== "z" && e.key.toLowerCase() !== "y") return;
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;   // let text fields keep their own undo
  e.preventDefault();
  const redo = (e.key.toLowerCase() === "z" && e.shiftKey) || e.key.toLowerCase() === "y";
  redo ? historyRedo() : historyUndo();
});

// Context-aware shortcuts (no modifiers, outside form fields): Del deletes the
// selected node / point, Enter closes a tissue region, Esc finishes, I toggles
// inlet/outlet, A/V pick artery/vein, G grows, R runs, 1–3 switch tabs.
const clickIfEnabled = (id) => { const b = $(id); if (b && !b.disabled) b.click(); };
const pickKind = (k) => {
  document.querySelector(`input[name=vtype][value=${k}]`).checked = true;
  if (state.editSel && state.editSel.kind !== k) deselect();
};

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;                // combos handled above / by the browser
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;   // don't hijack typing in fields
  const m = state.mode, k = e.key.toLowerCase();
  let handled = true;
  if (e.key === "Delete" || e.key === "Backspace") {
    if (m === "vessels") deleteSelectedNode();
    else if (m === "tissue") deleteSelection();
    else handled = false;
  } else if (e.key === "Enter") {
    if (m === "tissue" && state.tissueStroke.length) closePolygon();
    else handled = false;
  } else if (e.key === "Escape") {
    if (m === "vessels") deselect();
    else if (m === "tissue") cancelOrDeselect();
    else handled = false;
  } else if (k === "i" && m === "vessels") {
    toggleInlet();
  } else if (k === "a" && m === "vessels") {
    pickKind("artery");
  } else if (k === "v" && m === "vessels") {
    pickKind("vein");
  } else if (k === "g" && m === "vessels") {
    clickIfEnabled("btn-grow");
  } else if (k === "r") {
    clickIfEnabled("btn-run");                                   // run simulation
  } else if (e.key >= "1" && e.key <= "3") {
    setMode(MODES[+e.key - 1]);
  } else {
    handled = false;
  }
  if (handled) e.preventDefault();
});

// ── wiring ────────────────────────────────────────────────────────────────────
$("btn-new").addEventListener("click", guarded(newProject));
$("btn-frog").addEventListener("click", guarded(loadFrog));
$("btn-save-file").addEventListener("click", guarded(saveToFile));
$("btn-load-file").addEventListener("click", () => $("file-load").click());
$("file-load").addEventListener("change", guarded(loadFromFile));
$("btn-hist-undo").addEventListener("click", historyUndo);
$("btn-hist-redo").addEventListener("click", historyRedo);

// vessels
$("draw-radius").addEventListener("input", onRadiusInput);
$("draw-radius").addEventListener("change", recordHistory);   // one history entry per resize
$("radius-slider").addEventListener("input", onRadiusSlider);
$("radius-slider").addEventListener("change", recordHistory);
$("btn-seg-split").addEventListener("click", insertNodeOnSelected);
document.querySelectorAll("input[name=vtype]").forEach((r) =>
  r.addEventListener("change", () => pickKind(r.value)));
$("btn-node-io").addEventListener("click", toggleInlet);
$("btn-node-del").addEventListener("click", deleteSelectedNode);
$("btn-save-vessels").addEventListener("click", guarded(saveEdits));
$("btn-edit-revert").addEventListener("click", guarded(revertEdits));
$("btn-clear-kind").addEventListener("click", clearKind);
$("img-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) guarded(() => uploadImage(f))();
  e.target.value = "";
});

// tissue
$("tissue-target").addEventListener("change", () => {
  syncTargetRows();
  if (state.tissueSel && state.tissueSel.type !== $("tissue-target").value) {
    state.tissueSel = null; syncTissuePanel();
  }
  redraw();
});
$("lesion-k").addEventListener("input", onTissueValue);
$("hotspot-w").addEventListener("input", onTissueValue);
$("lesion-k").addEventListener("change", recordHistory);
$("hotspot-w").addEventListener("change", recordHistory);
$("btn-tissue-close").addEventListener("click", closePolygon);
$("btn-tissue-del").addEventListener("click", deleteSelection);
$("btn-save-mask").addEventListener("click", guarded(saveMask));
$("btn-tissue-revert").addEventListener("click", guarded(async () => {
  await loadMask(); recordHistory(); redraw(); status("reverted to saved tissue", "ok");
}));
$("btn-clear-tissue").addEventListener("click", clearTarget);

// grow / analyse / run
$("btn-grow").addEventListener("click", guarded(grow));
$("btn-prune").addEventListener("click", guarded(prune));
$("btn-adapt").addEventListener("click", guarded(adapt));
$("btn-morph-art").addEventListener("click", guarded(() => loadMorphometry("artery")));
$("btn-morph-ven").addEventListener("click", guarded(() => loadMorphometry("vein")));
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
MODES.forEach((m) => $(`mode-${m}`).addEventListener("click", () => setMode(m)));
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
$("btn-uq").addEventListener("click", guarded(runUQ));
$("btn-mc-perf").addEventListener("click", guarded(() => loadMulticompartment("perfusion")));
$("btn-mc-comp0").addEventListener("click", guarded(() => loadMulticompartment("arteriole")));
$("btn-mc-compN").addEventListener("click", guarded(() => loadMulticompartment("venule")));
$("tkm-model").addEventListener("change", () => { if (state.tracer) fitKinetics().catch(() => {}); });
$("flow-play").addEventListener("click", toggleFlowPlay);
$("flow-scrub").addEventListener("input", (e) => {
  pauseFlow(); state.flow.idx = Number(e.target.value); syncFlowUI(); redraw();
});

// warn before closing the tab with unsaved geometry
window.addEventListener("beforeunload", (e) => {
  if ($("btn-save-vessels").textContent.endsWith("•") || $("btn-save-mask").textContent.endsWith("•")) {
    e.preventDefault(); e.returnValue = "";
  }
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
    $("perm_y").value = g.permeability_y != null ? g.permeability_y / (g.permeability || 1) : 1;
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
  resetGeometry();
  state.hasResult = !!info.has_result;
  await loadVessels();
  if (await vectorizeLegacy()) state.hasResult = false;   // legacy strokes -> network
  await Promise.all([loadNetwork(), loadMask(), loadIO(), loadEditable()]);
  adoptLegacyIO();                           // legacy inlet/outlet points -> network nodes
  if (state.hasResult) {
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
