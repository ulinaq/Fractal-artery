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
  mode: "vessels",           // "vessels" | "tissue" | "result"
  field: "pressure",
  vessels: { arteries: [], veins: [] },   // saved strokes
  stroke: [],                // current in-progress vessel polyline (domain coords)
  regions: [],               // saved tissue outline polygons (domain coords)
  regionStroke: [],          // current in-progress tissue polygon
  inlets: [],                // arterial inlet points (domain coords)
  outlets: [],               // venous outlet points (domain coords)
  network: null,             // {segments:[...]} overlaid in result mode
  editable: null,            // {artery, vein} vessel networks being edited
  editSource: null,          // "network" once an editable vessel network is loaded
  editSel: null,             // {kind, node, seg} current selection
  editDrag: false,           // a node is being dragged
  editAdd: false,            // add-branch mode: clicks grow new edges
  fieldData: null,           // {name, shape, data, vmin, vmax}
  tracer: null,              // {shape, times, vmax, frames:[...]}
  flow: { active: false, idx: 0, timer: null },
  busy: false,               // a job is running -> lock actions
  hasResult: false,          // a solved result exists (enables Export)
};

// ── status helpers ────────────────────────────────────────────────────────────
const statusEl = document.getElementById("status");
function status(text, kind = "") { statusEl.textContent = text; statusEl.className = "status " + kind; }
const $ = (id) => document.getElementById(id);
const numOf = (id) => Number($(id).value);
const vtype = () => document.querySelector('input[name=vtype]:checked').value;

// ── undo / redo (workspace geometry history) ───────────────────────────────────
// Snapshots the editable geometry after each action — vessel network, hand-drawn
// strokes, tissue regions and inlet/outlet points — so any edit can be reverted.
const history = { stack: [], idx: -1 };

function geomSnapshot() {
  return JSON.stringify({
    vessels: state.vessels, regions: state.regions,
    inlets: state.inlets, outlets: state.outlets,
    editable: state.editable, editSource: state.editSource,
  });
}

function recordHistory() {
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
  "btn-grow", "btn-run", "btn-flow",
];

function refreshUI() {
  const ready = !!state.project && !state.busy;
  NEEDS_PROJECT.forEach((id) => { $(id).disabled = !ready; });
  $("btn-new").disabled = state.busy;
  $("btn-frog").disabled = state.busy;
  $("btn-load-file").disabled = state.busy;
  const canExport = !!(state.project && state.hasResult) && !state.busy;
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

function requireProject() {
  if (!state.project) { status("create a project first", "err"); throw new Error("no project"); }
  return state.project;
}

// ── interactive vessel editing (network / tree model) ──────────────────────────
// Vessels are a network — nodes + edges, each edge carrying its own radius (and,
// later, more per-edge attributes). The frog/segmented/grown trees already are
// networks; hand-drawn strokes are folded into one via POST /vectorize on entry.
let editDirty = false;

async function loadEditable() {
  state.editable = null; state.editSel = null; state.editAdd = false; state.editSource = null;
  clearEditDirty();
  if (!state.project) { if (state.mode === "edit") syncEditPanel(); return; }
  try {
    const e = await api("GET", `/projects/${state.project}/editable`);
    if (e && (e.artery || e.vein)) { state.editable = e; state.editSource = "network"; }
  } catch { /* no editable network */ }
  if (state.mode === "edit") syncEditPanel();
}

// On entering Edit with only hand-drawn strokes, convert them into a network.
async function ensureNetworkForEdit() {
  if (state.editSource || !state.project) return;
  if (!(state.vessels.arteries.length || state.vessels.veins.length)) return;
  try {
    await api("POST", `/projects/${state.project}/vectorize`);
    await Promise.all([loadVessels(), loadNetwork(), loadEditable()]);
    state.hasResult = false; refreshUI();
    syncEditPanel(); recordHistory(); redraw();
    status("hand-drawn vessels converted to an editable network", "ok");
  } catch (e) { status(e.message, "err"); }
}

function hasEditable() { return !!(state.editable && (state.editable.artery || state.editable.vein)); }
function editKinds() { return ["artery", "vein"].filter((k) => state.editable && state.editable[k]); }

// squared distance from a point to a segment (canvas px) + the param t along it
function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  const cx = ax + t * dx, cy = ay + t * dy;
  return { d: (px - cx) ** 2 + (py - cy) ** 2, t };
}

// nearest node handle to a canvas point. Tries node handles first (tol px); if
// none, falls back to clicking *near a vessel edge* -> its nearer endpoint, so a
// dense network (e.g. the frog) is easy to grab without pixel-hunting the dot.
function pickNode(px, py, tol = 13) {
  let best = null, bestD = tol * tol;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    for (let i = 0; i < n.node_x.length; i++) {
      const [cx, cy] = toCanvas(n.node_x[i], n.node_y[i]);
      const d = (cx - px) ** 2 + (cy - py) ** 2;
      if (d < bestD) { bestD = d; best = { kind, node: i }; }
    }
  }
  if (best || tol > 500) return best;         // huge tol (add-mode) skips the edge fallback
  let edgeD = 11 * 11, pick = null;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    for (let s = 0; s < n.seg_start.length; s++) {
      const [ax, ay] = toCanvas(n.node_x[n.seg_start[s]], n.node_y[n.seg_start[s]]);
      const [bx, by] = toCanvas(n.node_x[n.seg_end[s]], n.node_y[n.seg_end[s]]);
      const { d, t } = segDist2(px, py, ax, ay, bx, by);
      if (d < edgeD) { edgeD = d; pick = { kind, node: t < 0.5 ? n.seg_start[s] : n.seg_end[s] }; }
    }
  }
  return pick;
}

// the edge to resize for a node: the one feeding it, else an outgoing one
function nodeSegment(net, node) {
  const s = net.seg_end.indexOf(node);
  return s >= 0 ? s : net.seg_start.indexOf(node);
}

function selectNode(hit) {
  hit.seg = nodeSegment(state.editable[hit.kind], hit.node);
  state.editSel = hit;
}

function editPointerDown(px, py) {
  if (!hasEditable()) return;
  if (state.editAdd) { editAddAt(px, py); return; }
  const hit = pickNode(px, py);
  if (hit) { selectNode(hit); state.editDrag = true; } else { state.editSel = null; }
  syncEditPanel(); redraw();
}

function moveNode(kind, node, x, y) {
  const n = state.editable[kind];
  n.node_x[node] = x; n.node_y[node] = y;
}

// add-branch mode: each click grows a new edge *from the selected node* to the
// click point. The source node stays selected, so clicking several times fans
// out multiple children — turning it into a bifurcation (add to a node that
// already has an edge and it becomes a branch point).
function editAddAt(px, py) {
  const [x, y] = toDomain(px, py);
  const sel = state.editSel;
  if (!sel) { const hit = pickNode(px, py, 1e9); if (!hit) return; selectNode(hit); syncEditPanel(); redraw(); return; }
  const kind = sel.kind, n = state.editable[kind], ni = n.node_x.length;
  n.node_x.push(x); n.node_y.push(y);
  n.seg_start.push(sel.node); n.seg_end.push(ni);           // new child edge from the SAME node
  n.seg_radius.push(sel.seg >= 0 ? n.seg_radius[sel.seg] : 2);   // inherit the parent edge's radius
  selectNode({ kind, node: sel.node });                     // keep the source selected to keep branching
  markEditDirty(); recordHistory(); syncEditPanel(); redraw();
}

function deleteSelectedNode() {
  if (state.busy) return;
  const sel = state.editSel;
  if (!sel) return;
  const n = state.editable[sel.kind];
  if ((n.inlet_nodes || []).includes(sel.node)) { status("can't delete an inlet/outlet node", "err"); return; }
  const i = sel.node;
  const keep = n.seg_start.map((s0, k) => s0 !== i && n.seg_end[k] !== i);
  const sh = (v) => (v > i ? v - 1 : v);
  n.seg_start = n.seg_start.filter((_, k) => keep[k]).map(sh);
  n.seg_end = n.seg_end.filter((_, k) => keep[k]).map(sh);
  n.seg_radius = n.seg_radius.filter((_, k) => keep[k]);
  n.node_x.splice(i, 1); n.node_y.splice(i, 1);
  n.inlet_nodes = (n.inlet_nodes || []).filter((x) => x !== i).map(sh);
  if (n.root_node != null && n.root_node !== i) n.root_node = sh(n.root_node);
  pruneNetworkKind(n);          // drop nodes orphaned to degree 0 by the deletion
  state.editSel = null; markEditDirty(); recordHistory(); syncEditPanel(); redraw();
}

// drop degree-0 non-inlet nodes, reindex (mirrors the backend rebuild)
function pruneNetworkKind(n) {
  const N = n.node_x.length;
  const deg = new Array(N).fill(0);
  n.seg_start.forEach((s, k) => { deg[s]++; deg[n.seg_end[k]]++; });
  const inl = new Set(n.inlet_nodes || []);
  const keep = Array.from({ length: N }, (_, i) => deg[i] > 0 || inl.has(i));
  const o2n = new Array(N).fill(-1); let j = 0;
  for (let i = 0; i < N; i++) if (keep[i]) o2n[i] = j++;
  n.node_x = n.node_x.filter((_, i) => keep[i]);
  n.node_y = n.node_y.filter((_, i) => keep[i]);
  n.seg_start = n.seg_start.map((v) => o2n[v]);
  n.seg_end = n.seg_end.map((v) => o2n[v]);
  n.inlet_nodes = (n.inlet_nodes || []).filter((i) => keep[i]).map((v) => o2n[v]);
  if (n.root_node != null) n.root_node = keep[n.root_node] ? o2n[n.root_node] : (n.inlet_nodes[0] ?? null);
}

function toggleAddMode() {
  if (!hasEditable()) return;
  state.editAdd = !state.editAdd;
  $("btn-edit-add").classList.toggle("active", state.editAdd);
  canvas.style.cursor = state.editAdd ? "copy" : "pointer";
  status(state.editAdd ? "add mode: click to grow a new edge from the selected node" : "");
}

function syncEditPanel() {
  const sl = $("edit-radius"), sel = state.editSel, can = hasEditable();
  $("btn-edit-add").disabled = !can;
  $("btn-edit-del").disabled = !(sel && can && !(state.editable[sel.kind].inlet_nodes || []).includes(sel.node));
  if (sel && sel.seg >= 0) {
    const r = state.editable[sel.kind].seg_radius[sel.seg];
    sl.value = r; sl.disabled = false;
    $("edit-radius-val").textContent = (+r).toFixed(1);
    $("edit-sel").textContent = `${sel.kind} node #${sel.node} — drag to move, slider sets this edge's radius.`;
  } else {
    sl.disabled = true; $("edit-radius-val").textContent = "";
    $("edit-sel").textContent = can
      ? "Vessel network — click a node to select it."
      : "No vessels yet — draw, segment or grow a tree first.";
  }
}

function onEditRadius() {
  const sel = state.editSel;
  if (!sel || sel.seg < 0) return;
  const r = numOf("edit-radius");
  state.editable[sel.kind].seg_radius[sel.seg] = r;     // per-edge radius
  $("edit-radius-val").textContent = r.toFixed(1);
  markEditDirty(); redraw();
}

function markEditDirty() { editDirty = true; $("btn-edit-save").textContent = "Save edits •"; }
function clearEditDirty() { editDirty = false; if ($("btn-edit-save")) $("btn-edit-save").textContent = "Save edits"; }

async function saveEdits() {
  requireProject();
  if (state.editSource !== "network") { status("nothing to edit", "err"); return; }
  const pack = (k) => {
    const n = state.editable[k];
    return n ? { node_x: n.node_x, node_y: n.node_y, seg_radius: n.seg_radius,
                 seg_start: n.seg_start, seg_end: n.seg_end,
                 inlet_nodes: n.inlet_nodes || null, root_node: n.root_node ?? null } : null;
  };
  await api("PUT", `/projects/${state.project}/editable`,
            { artery: pack("artery"), vein: pack("vein") });
  state.hasResult = false;                       // geometry changed -> result stale
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  syncEditPanel(); refreshUI(); recordHistory(); redraw();
  status("vessel edits saved", "ok");
}

async function revertEdits() {
  state.editAdd = false; $("btn-edit-add").classList.remove("active");
  canvas.style.cursor = "pointer";
  await loadEditable();
  syncEditPanel(); redraw();
  status("reverted to saved geometry", "ok");
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
  else if (state.mode === "tissue") state.regionStroke.push(pt);
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

async function saveVessels() {
  requireProject();
  await api("PUT", `/projects/${state.project}/vessels`, state.vessels);
  $("use-vessels").checked = true;
  status(`saved ${state.vessels.arteries.length} arteries, ${state.vessels.veins.length} veins`, "ok");
}

// ── tissue mask drawing ────────────────────────────────────────────────────────
function endRegion() {
  if (state.regionStroke.length >= 3) {
    state.regions.push(state.regionStroke.map(([x, y]) => [x, y]));
    recordHistory();
  }
  state.regionStroke = [];
  redraw();
}
function undoRegion() {
  if (state.regionStroke.length) state.regionStroke.pop();
  else { state.regions.pop(); recordHistory(); }
  redraw();
}
function clearTissue() { state.regions = []; state.regionStroke = []; recordHistory(); redraw(); }

async function saveMask() {
  requireProject();
  if (state.regionStroke.length >= 3) endRegion();   // auto-close a pending outline
  await api("PUT", `/projects/${state.project}/mask`, { regions: state.regions });
  status(state.regions.length ? `saved tissue (${state.regions.length} region(s))`
                              : "tissue cleared — full domain active", "ok");
}

async function loadMask() {
  try { state.regions = (await api("GET", `/projects/${state.project}/mask`)).regions || []; }
  catch { state.regions = []; }
}

async function loadVessels() {
  try {
    const v = await api("GET", `/projects/${state.project}/vessels`);
    state.vessels = { arteries: v.arteries || [], veins: v.veins || [] };
  } catch { state.vessels = { arteries: [], veins: [] }; }
}

// ── inlets / outlets ──────────────────────────────────────────────────────────
function clearIO() { state.inlets = []; state.outlets = []; recordHistory(); redraw(); }

async function saveIO() {
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

// ── grow / run ─────────────────────────────────────────────────────────────────
async function grow() {
  requireProject();
  const target = document.querySelector("input[name=growtarget]:checked").value;
  if (state.vessels.arteries.length || state.vessels.veins.length) await saveVessels();
  if (state.regions.length || state.regionStroke.length >= 3) await saveMask();
  status(`growing ${target}…`);
  const info = await api("POST", `/projects/${state.project}/grow`,
    { target, method: $("grow-method").value, n_points: numOf("grow-npts"),
      step: numOf("grow-step"), seed: numOf("seed") });
  $("grow-info").textContent = `grown ${target}: ${info.n_segments} segments, ${info.n_terminals} terminals`;
  state.hasResult = false;         // grow invalidates any prior result server-side
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  recordHistory();
  status(`${target} tree grown`, "ok");
  redraw();
}

async function run() {
  requireProject();
  if (state.vessels.arteries.length || state.vessels.veins.length) await saveVessels();
  if (state.regions.length || state.regionStroke.length >= 3) await saveMask();
  if (state.inlets.length || state.outlets.length) await saveIO();
  const job = await api("POST", `/projects/${state.project}/run`,
    { nonlinear: $("nonlinear").checked, use_vessels: $("use-vessels").checked });
  status("running…");
  $("run-info").classList.remove("run-error");
  $("run-info").textContent = "job " + job.id;
  const final = await pollJob(job.id);
  if (final.state === "error") {
    status("run failed — see details below", "err");
    $("run-info").classList.add("run-error");
    $("run-info").textContent = final.error || "the run failed without an error message";
    return;
  }
  $("run-info").classList.remove("run-error");
  status("solved", "ok");
  state.hasResult = true;
  await Promise.all([loadNetwork(), showSummary(), loadField(state.field)]);
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

// ── tracer flow video ─────────────────────────────────────────────────────────
function stopTimer() {
  if (state.flow.timer) { clearInterval(state.flow.timer); state.flow.timer = null; }
}

async function computeFlow() {
  requireProject();
  if (!state.tracer) {
    try {
      await loadTracer();                 // reuse a previously computed tracer
    } catch {
      status("computing flow… (this solves + advects a bolus)");
      const job = await api("POST", `/projects/${state.project}/tracer`, { n_frames: 48 });
      const final = await pollJob(job.id);
      if (final.state === "error") { status(final.error || "flow failed", "err"); return; }
      await loadTracer();
      status("flow ready", "ok");
    }
  }
  state.flow.active = true;
  ["flow-play", "flow-scrub", "flow-time", "flow-gif"].forEach((id) => ($(id).hidden = false));
  playFlow();
}

async function loadTracer() {
  const t = await api("GET", `/projects/${state.project}/tracer`);
  state.tracer = t;
  state.flow.idx = 0;
  $("flow-scrub").max = t.n_frames - 1;
  $("flow-gif").href = `/projects/${state.project}/tracer.gif`;
}

function playFlow() {
  stopTimer();
  state.flow.timer = setInterval(() => {
    state.flow.idx = (state.flow.idx + 1) % state.tracer.n_frames;
    syncFlowUI(); redraw();
  }, 90);
  $("flow-play").textContent = "⏸";
}
function pauseFlow() { stopTimer(); $("flow-play").textContent = "▶"; }
function toggleFlowPlay() { state.flow.timer ? pauseFlow() : playFlow(); }
function stopFlow() {
  pauseFlow(); state.flow.active = false;
  ["flow-play", "flow-scrub", "flow-time"].forEach((id) => ($(id).hidden = true));
}
function syncFlowUI() {
  $("flow-scrub").value = state.flow.idx;
  $("flow-time").textContent = `t = ${state.tracer.times[state.flow.idx].toFixed(2)}`;
}

// ── rendering ──────────────────────────────────────────────────────────────────
function redraw() {
  ctx.clearRect(0, 0, W, H);
  if (state.mode === "result") {
    if (state.flow.active && state.tracer) drawTracerFrame(state.flow.idx);
    else if (state.fieldData) drawField();
  }
  drawRegions();
  drawNetworkOverlay();
  drawEdit();
  drawVessels();
  drawStroke();
  drawIO();
}

function drawIO() {
  // inlets ▲ (arterial, red), outlets ▼ (venous, blue); shown in I/O and Result
  if (state.mode !== "io" && state.mode !== "result") return;
  const marker = (x, y, up, color) => {
    const [px, py] = toCanvas(x, y), s = 7;
    ctx.beginPath();
    if (up) { ctx.moveTo(px, py - s); ctx.lineTo(px - s, py + s); ctx.lineTo(px + s, py + s); }
    else { ctx.moveTo(px, py + s); ctx.lineTo(px - s, py - s); ctx.lineTo(px + s, py - s); }
    ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
  };
  state.inlets.forEach(([x, y]) => marker(x, y, true, getCss("--art")));
  state.outlets.forEach(([x, y]) => marker(x, y, false, getCss("--ven")));
}

function tracePath(points) {
  points.forEach(([x, y], i) => {
    const [px, py] = toCanvas(x, y);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
}

function drawRegions() {
  // saved tissue outlines: translucent fill (skip under a rendered field)
  const showFill = !(state.mode === "result" && state.fieldData);
  for (const poly of state.regions) {
    ctx.beginPath(); tracePath(poly); ctx.closePath();
    if (showFill) { ctx.fillStyle = "rgba(16,185,129,.12)"; ctx.fill(); }
    ctx.strokeStyle = "rgba(5,150,105,.7)"; ctx.lineWidth = 1.5; ctx.stroke();
  }
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
}

function drawField() {
  const { shape, data, vmin, vmax } = state.fieldData;
  paintHeatmap(data, shape[0], shape[1], vmin, vmax);
  updateColorbar(vmin, vmax, state.field);
}

function drawTracerFrame(idx) {
  const t = state.tracer;
  paintHeatmap(t.frames[idx], t.shape[0], t.shape[1], 0, t.vmax);
  updateColorbar(0, t.vmax, "tracer");
}

function drawEdit() {
  if (state.mode !== "edit" || !hasEditable()) return;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    const color = kind === "vein" ? getCss("--ven") : getCss("--art");
    ctx.globalAlpha = 0.55;
    for (let s = 0; s < n.seg_start.length; s++) {
      const [x0, y0] = toCanvas(n.node_x[n.seg_start[s]], n.node_y[n.seg_start[s]]);
      const [x1, y1] = toCanvas(n.node_x[n.seg_end[s]], n.node_y[n.seg_end[s]]);
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(1, n.seg_radius[s] * 0.6);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < n.node_x.length; i++) {
      const [cx, cy] = toCanvas(n.node_x[i], n.node_y[i]);
      const sel = state.editSel && state.editSel.kind === kind && state.editSel.node === i;
      ctx.beginPath(); ctx.arc(cx, cy, sel ? 6 : 3.2, 0, 7);
      ctx.fillStyle = sel ? getCss("--accent") : "#fff";
      ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = color; ctx.stroke();
    }
  }
}

function drawNetworkOverlay() {
  if (!state.network || state.mode === "edit") return;    // edit mode draws the editable tree instead
  for (const s of state.network.segments) {
    const [x0, y0] = toCanvas(s.x0, s.y0), [x1, y1] = toCanvas(s.x1, s.y1);
    ctx.strokeStyle = s.kind === "vein" ? "rgba(191,219,254,.95)" : "rgba(254,202,202,.95)";
    ctx.lineWidth = Math.max(0.6, s.r * 0.5);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
}

function drawVessels() {
  if (state.mode === "edit") return;      // edit mode draws the editable copy instead
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

function markVertices(points) {
  for (const [x, y] of points) {
    const [px, py] = toCanvas(x, y);
    ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill();
  }
}

function drawStroke() {
  if (state.mode === "vessels" && state.stroke.length) {
    ctx.strokeStyle = vtype() === "artery" ? getCss("--art") : getCss("--ven");
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = Math.max(1.5, numOf("draw-radius") * 1.2);
    ctx.beginPath(); tracePath(state.stroke); ctx.stroke();
    markVertices(state.stroke);
  } else if (state.mode === "tissue" && state.regionStroke.length) {
    ctx.strokeStyle = "rgba(5,150,105,.9)"; ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1.8;
    ctx.beginPath(); tracePath(state.regionStroke); ctx.stroke();
    markVertices(state.regionStroke);
  }
}

function updateColorbar(vmin, vmax, label) {
  const cb = $("colorbar"); cb.hidden = false;
  const stops = VIRIDIS.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i / (VIRIDIS.length - 1)).toFixed(0)}%`);
  $("cb-ramp").style.background = `linear-gradient(to top, ${stops.join(",")})`;
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(1) : v.toFixed(2);
  $("cb-max").textContent = fmt(vmax); $("cb-min").textContent = fmt(vmin);
  $("cb-title").textContent = label;
}

const getCss = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ── modes ────────────────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  $("mode-vessels").classList.toggle("active", mode === "vessels");
  $("mode-tissue").classList.toggle("active", mode === "tissue");
  $("mode-io").classList.toggle("active", mode === "io");
  $("mode-edit").classList.toggle("active", mode === "edit");
  $("mode-result").classList.toggle("active", mode === "result");
  $("field-picker").classList.toggle("show", mode === "result");
  $("flowbar").classList.toggle("show", mode === "result");
  $("colorbar").hidden = mode !== "result";
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
$("btn-save-mask").addEventListener("click", () => saveMask().catch((e) => status(e.message, "err")));
$("btn-grow").addEventListener("click", guarded(grow));
$("btn-run").addEventListener("click", guarded(run));
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
    await Promise.all([showSummary(), loadField(state.field)]);
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
