// perfusion 2-D workspace — interactive vessel editing (the network / tree model).
// Vessels are a network: nodes + edges, each edge carrying its own radius (and,
// later, more per-edge attributes). The frog/segmented/grown trees already are
// networks; hand-drawn strokes are folded into one via POST /vectorize on entry.
//
// This module and app.js reference each other (edit calls recordHistory/refreshUI/
// loaders; app wires the edit handlers) — a cycle resolved by ES-module live
// bindings, since every cross-call happens at runtime, not at module load.
import {
  api, canvas, editKinds, hasEditable, numOf, state, status, toCanvas, toDomain, $,
} from "./core.js";
import { redraw } from "./render.js";
import { loadNetwork, loadVessels, recordHistory, refreshUI, requireProject, stopFlow } from "./app.js";

let editDirty = false;

export async function loadEditable() {
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
export async function ensureNetworkForEdit() {
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

export function editPointerDown(px, py) {
  if (!hasEditable()) return;
  if (state.editAdd) { editAddAt(px, py); return; }
  const hit = pickNode(px, py);
  if (hit) { selectNode(hit); state.editDrag = true; } else { state.editSel = null; }
  syncEditPanel(); redraw();
}

export function moveNode(kind, node, x, y) {
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

export function deleteSelectedNode() {
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

export function toggleAddMode() {
  if (!hasEditable()) return;
  state.editAdd = !state.editAdd;
  $("btn-edit-add").classList.toggle("active", state.editAdd);
  canvas.style.cursor = state.editAdd ? "copy" : "pointer";
  status(state.editAdd ? "add mode: click to grow a new edge from the selected node" : "");
}

export function syncEditPanel() {
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

export function onEditRadius() {
  const sel = state.editSel;
  if (!sel || sel.seg < 0) return;
  const r = numOf("edit-radius");
  state.editable[sel.kind].seg_radius[sel.seg] = r;     // per-edge radius
  $("edit-radius-val").textContent = r.toFixed(1);
  markEditDirty(); redraw();
}

export function markEditDirty() { editDirty = true; $("btn-edit-save").textContent = "Save edits •"; }
export function clearEditDirty() { editDirty = false; if ($("btn-edit-save")) $("btn-edit-save").textContent = "Save edits"; }

export async function saveEdits() {
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

export async function revertEdits() {
  state.editAdd = false; $("btn-edit-add").classList.remove("active");
  canvas.style.cursor = "pointer";
  await loadEditable();
  syncEditPanel(); redraw();
  status("reverted to saved geometry", "ok");
}
