// perfusion 2-D workspace — the vessel network editor (Vessels tab).
//
// Vessels are a network: nodes + edges, each edge carrying its own radius. The
// canvas edits that network directly — there is no separate "draw" vs "edit":
//
//   click empty canvas      → start a new vessel (its first node is an inlet/outlet)
//   click again             → extend: a new edge from the selected node to the click
//   click a node            → select it (then click empty space to branch from it)
//   drag a node             → move it
//   click on a vessel       → select that segment: the radius box edits it, Del removes it
//   double-click a vessel   → insert a node there (split the edge) and select it
//   Shift+click node/vessel → connect the selected node to it (junction / loop)
//   Del                     → delete the selected node (selects its neighbour) / segment
//   Esc / click selected    → finish (deselect)
//
// The selection is a node {kind, node, seg: its edge} or a segment {kind, node: -1, seg}.
//   I                       → toggle the selected node as inlet (artery) / outlet (vein)
//
// Inlets/outlets are the network's `inlet_nodes` (prescribed-pressure nodes):
// p_in on the artery, p_out on the vein. Frog/segmented/grown trees are already
// networks; legacy hand-drawn polylines are folded into one via POST /vectorize.
//
// This module and app.js reference each other (edit calls recordHistory/refreshUI/
// loaders; app wires the handlers) — a cycle resolved by ES-module live bindings,
// since every cross-call happens at runtime, not at module load.
import { api, canvas, editKinds, numOf, state, status, toCanvas, toDomain, $ } from "./core.js";
import { redraw } from "./render.js";
import { loadVessels, recordHistory, refreshUI, requireProject } from "./app.js";
import { loadNetwork, stopFlow } from "./run.js";

const NODE_TOL = 11;          // px: pick radius for a node handle
const EDGE_TOL = 7;           // px: pick distance for a vessel edge
const DRAG_START = 4;         // px: pointer travel before a press becomes a drag
let editDirty = false;
let press = null;             // {px, py, kind, node, moved} while the pointer is down

const emptyNet = () => ({ node_x: [], node_y: [], seg_start: [], seg_end: [],
                          seg_radius: [], inlet_nodes: [], root_node: null });
export const newKind = () => document.querySelector("input[name=vtype]:checked").value;
const setKindRadio = (kind) => {
  const r = document.querySelector(`input[name=vtype][value=${kind}]`);
  if (r) r.checked = true;
};

// ── loading ───────────────────────────────────────────────────────────────────
export async function loadEditable() {
  state.editable = { artery: null, vein: null };
  state.editSel = null;
  clearEditDirty();
  if (!state.project) { syncEditPanel(); return; }
  try {
    const e = await api("GET", `/projects/${state.project}/editable`);
    if (e) state.editable = { artery: e.artery || null, vein: e.vein || null };
  } catch { /* no editable network */ }
  syncEditPanel();
}

// Legacy projects may hold hand-drawn polylines; fold them into the network once.
// Returns true when it converted something (the server then clears the result).
export async function vectorizeLegacy() {
  if (!state.project || !(state.vessels.arteries.length || state.vessels.veins.length)) return false;
  try {
    await api("POST", `/projects/${state.project}/vectorize`);
    await loadVessels();
    return true;
  } catch (e) { status(e.message, "err"); return false; }
}

// Legacy projects may hold inlet/outlet *points*; snap them onto network nodes
// (nearest leaf, like the server did) so they become the editable inlet_nodes.
export function adoptLegacyIO() {
  const pairs = [["artery", state.inlets], ["vein", state.outlets]];
  let adopted = false;
  for (const [kind, pts] of pairs) {
    const n = state.editable[kind];
    if (!n || !pts.length) continue;
    const deg = degrees(n);
    const ends = n.node_x.map((_, i) => i).filter((i) => deg[i] === 1 || (n.inlet_nodes || []).includes(i));
    const picked = new Set();
    for (const [x, y] of pts) {
      let best = -1, bd = Infinity;
      for (const i of ends) {
        const d = (n.node_x[i] - x) ** 2 + (n.node_y[i] - y) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
      if (best >= 0) picked.add(best);
    }
    // keep every connected component fed (as the server's snap_io did)
    const comp = components(n), covered = new Set([...picked].map((i) => comp[i]));
    for (const i of n.inlet_nodes || []) if (!covered.has(comp[i])) { picked.add(i); covered.add(comp[i]); }
    n.inlet_nodes = [...picked].sort((a, b) => a - b);
    adopted = true;
  }
  if (adopted) { state.inlets = []; state.outlets = []; state.legacyIO = true; markEditDirty(); }
}

// ── graph helpers ─────────────────────────────────────────────────────────────
function degrees(n) {
  const d = new Array(n.node_x.length).fill(0);
  n.seg_start.forEach((s, k) => { d[s]++; d[n.seg_end[k]]++; });
  return d;
}

function components(n) {
  const N = n.node_x.length, parent = Array.from({ length: N }, (_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  n.seg_start.forEach((s, k) => { parent[find(s)] = find(n.seg_end[k]); });
  return parent.map((_, i) => find(i));
}

const neighbours = (n, i) => n.seg_start.flatMap((s, k) =>
  (s === i ? [n.seg_end[k]] : n.seg_end[k] === i ? [s] : []));

// the edge whose radius a node "owns": the one feeding it, else an outgoing one
function nodeSegment(n, node) {
  const s = n.seg_end.indexOf(node);
  return s >= 0 ? s : n.seg_start.indexOf(node);
}

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  const cx = ax + t * dx, cy = ay + t * dy;
  return { d: (px - cx) ** 2 + (py - cy) ** 2, t };
}

function pickNode(px, py) {
  let best = null, bestD = NODE_TOL * NODE_TOL;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    for (let i = 0; i < n.node_x.length; i++) {
      const [cx, cy] = toCanvas(n.node_x[i], n.node_y[i]);
      const d = (cx - px) ** 2 + (cy - py) ** 2;
      if (d < bestD) { bestD = d; best = { kind, node: i }; }
    }
  }
  return best;
}

function pickEdge(px, py) {
  let best = null, bestD = EDGE_TOL * EDGE_TOL;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    for (let s = 0; s < n.seg_start.length; s++) {
      const [ax, ay] = toCanvas(n.node_x[n.seg_start[s]], n.node_y[n.seg_start[s]]);
      const [bx, by] = toCanvas(n.node_x[n.seg_end[s]], n.node_y[n.seg_end[s]]);
      const { d, t } = segDist2(px, py, ax, ay, bx, by);
      if (d < bestD) { bestD = d; best = { kind, seg: s, t }; }
    }
  }
  return best;
}

// ── mutations ─────────────────────────────────────────────────────────────────
function ensureNet(kind) {
  if (!state.editable[kind]) state.editable[kind] = emptyNet();
  return state.editable[kind];
}

function addNode(kind, x, y) {
  const n = ensureNet(kind);
  n.node_x.push(x); n.node_y.push(y);
  return n.node_x.length - 1;
}

function addEdge(kind, a, b, r) {
  const n = state.editable[kind];
  if (a === b) return;
  if (n.seg_start.some((s, k) => (s === a && n.seg_end[k] === b) || (s === b && n.seg_end[k] === a))) return;
  n.seg_start.push(a); n.seg_end.push(b); n.seg_radius.push(Math.max(1e-3, r));
}

// split an edge at parameter t (canvas-space projection) → index of the new node
function splitEdge(kind, seg, t) {
  const n = state.editable[kind];
  const a = n.seg_start[seg], b = n.seg_end[seg], r = n.seg_radius[seg];
  const m = addNode(kind, n.node_x[a] + t * (n.node_x[b] - n.node_x[a]),
                    n.node_y[a] + t * (n.node_y[b] - n.node_y[a]));
  n.seg_end[seg] = m;                         // a → m keeps the edge's id (and direction)
  n.seg_start.push(m); n.seg_end.push(b); n.seg_radius.push(r);
  return m;
}

function select(kind, node) {
  state.editSel = kind == null ? null : { kind, node, seg: nodeSegment(state.editable[kind], node) };
  if (kind != null) showSelRadius();
}

function selectSeg(kind, seg) {
  state.editSel = { kind, node: -1, seg };
  showSelRadius();
}

// reflect the selection's kind + edge radius in the panel controls
function showSelRadius() {
  const { kind, seg } = state.editSel;
  setKindRadio(kind);
  if (seg >= 0) setRadiusInputs(state.editable[kind].seg_radius[seg]);
}

function setRadiusInputs(r) {
  $("draw-radius").value = +(+r).toFixed(2);
  $("radius-slider").value = r;
}

// A lone node (a vessel start never extended) is dropped when you move on.
function dropIfIsolated(sel) {
  if (!sel || sel.node < 0) return false;
  const n = state.editable[sel.kind];
  if (n && sel.node < n.node_x.length && degrees(n)[sel.node] === 0) {
    removeNode(sel.kind, sel.node);
    return true;
  }
  return false;
}

function removeNode(kind, i) {
  const n = state.editable[kind];
  const keep = n.seg_start.map((s0, k) => s0 !== i && n.seg_end[k] !== i);
  const sh = (v) => (v > i ? v - 1 : v);
  n.seg_start = n.seg_start.filter((_, k) => keep[k]).map(sh);
  n.seg_end = n.seg_end.filter((_, k) => keep[k]).map(sh);
  n.seg_radius = n.seg_radius.filter((_, k) => keep[k]);
  n.node_x.splice(i, 1); n.node_y.splice(i, 1);
  n.inlet_nodes = (n.inlet_nodes || []).filter((x) => x !== i).map(sh);
  if (n.root_node != null) n.root_node = n.root_node === i ? (n.inlet_nodes[0] ?? null) : sh(n.root_node);
  if (!n.node_x.length) state.editable[kind] = null;
}

// drop degree-0 nodes and reindex (before saving; mirrors the backend rebuild)
function pruneIsolated(kind) {
  const n = state.editable[kind];
  if (!n) return;
  const deg = degrees(n);
  for (let i = n.node_x.length - 1; i >= 0; i--) {
    if (deg[i] === 0) removeNode(kind, i);
    if (!state.editable[kind]) return;
  }
}

// ── pointer interaction ───────────────────────────────────────────────────────
export function editPointerDown(px, py, e) {
  if (!state.project) { status("create a project first", "err"); return; }
  const prev = state.editSel;
  const shift = e && e.shiftKey;
  const hitN = pickNode(px, py);
  press = { px, py, moved: false, kind: null, node: -1 };

  if (hitN) {
    if (shift && prev && prev.node >= 0 && prev.kind === hitN.kind && prev.node !== hitN.node) {
      // connect the selected node to an existing one (junction / loop)
      addEdge(hitN.kind, prev.node, hitN.node, numOf("draw-radius"));
      press = null; select(null); changed("connected");
      return;
    }
    if (prev && prev.kind === hitN.kind && prev.node === hitN.node) {
      press = { ...press, kind: hitN.kind, node: hitN.node, toggleOff: true };
    } else {
      let node = hitN.node;
      if (dropIfIsolated(prev) && prev.kind === hitN.kind && prev.node < node) node--;   // reindexed
      select(hitN.kind, node);
      press = { ...press, kind: hitN.kind, node };
    }
    syncEditPanel(); redraw();
    return;
  }

  const hitE = pickEdge(px, py);
  if (hitE) {
    press = null;
    if (shift && prev && prev.node >= 0 && prev.kind === hitE.kind) {
      const m = splitEdge(hitE.kind, hitE.seg, hitE.t);
      addEdge(hitE.kind, prev.node, m, numOf("draw-radius"));     // branch ends on the vessel
      select(null); changed("joined onto vessel");
      return;
    }
    if (!(prev && prev.node < 0 && prev.kind === hitE.kind && prev.seg === hitE.seg)) {
      dropIfIsolated(prev);
      const again = pickEdge(px, py);                // indices may shift if a lone node went
      selectSeg(again.kind, again.seg);
    }
    syncEditPanel(); redraw();
    return;
  }

  // empty canvas
  const [x, y] = toDomain(px, py);
  if (prev && prev.node < 0) {                        // a segment was selected: just deselect
    select(null); syncEditPanel(); redraw();
    return;
  }
  if (prev) {
    // extend: a new edge from the selected node; the tip becomes the selection
    const tip = addNode(prev.kind, x, y);
    addEdge(prev.kind, prev.node, tip, numOf("draw-radius"));
    select(prev.kind, tip);
    changed();
  } else {
    // start a new vessel; its first node is the feed (inlet for artery, outlet for vein)
    const kind = newKind();
    const i = addNode(kind, x, y);
    const n = state.editable[kind];
    n.inlet_nodes = [...(n.inlet_nodes || []), i];
    if (n.root_node == null) n.root_node = i;
    select(kind, i);
    syncEditPanel(); redraw();
    status(`new ${kind}: click to add points · click a node to branch from it · Esc to finish`);
  }
}

export function editPointerMove(px, py) {
  if (press && press.node >= 0) {
    if (!press.moved && Math.hypot(px - press.px, py - press.py) < DRAG_START) return;
    press.moved = true;
    const [x, y] = toDomain(px, py);
    const n = state.editable[press.kind];
    n.node_x[press.node] = x; n.node_y[press.node] = y;
    redraw();
    return;
  }
  state.hover = toDomain(px, py);        // rubber-band preview from the selection
  canvas.style.cursor = pickNode(px, py) ? "move" : pickEdge(px, py) ? "pointer" : "crosshair";
  if (state.editSel && state.editSel.node >= 0) redraw();
}

// double-click a vessel: insert a node there and select it (to branch from it)
export function editDoubleClick(px, py) {
  const hitE = pickEdge(px, py);
  if (!hitE || pickNode(px, py)) return;
  const m = splitEdge(hitE.kind, hitE.seg, hitE.t);
  select(hitE.kind, m);
  changed("node inserted — click empty space to branch from it");
}

export function insertNodeOnSelected() {
  const sel = state.editSel;
  if (!sel || sel.node >= 0) return;
  const m = splitEdge(sel.kind, sel.seg, 0.5);
  select(sel.kind, m);
  changed("node inserted at the segment's midpoint");
}

function deleteSegment(kind, seg) {
  const n = state.editable[kind];
  const ends = [n.seg_start[seg], n.seg_end[seg]];
  const inl = n.inlet_nodes || [];
  n.seg_start.splice(seg, 1); n.seg_end.splice(seg, 1); n.seg_radius.splice(seg, 1);
  // drop endpoints left with no segment (higher index first so the other stays valid)
  const deg = degrees(n);
  const lone = ends.filter((i) => deg[i] === 0).sort((a, b) => b - a);
  const survivor = ends.find((i) => deg[i] > 0);
  const lostInlet = lone.some((i) => inl.includes(i));
  lone.forEach((i) => { if (state.editable[kind]) removeNode(kind, i); });
  const m = state.editable[kind];
  if (m && lostInlet && !(m.inlet_nodes || []).length && survivor != null) {
    const heir = survivor - lone.filter((i) => i < survivor).length;   // reindexed
    m.inlet_nodes = [heir]; m.root_node = heir;                         // keep the tree fed
  }
}

export function editPointerUp() {
  if (!press) return;
  const p = press; press = null;
  if (p.moved) { changed(); return; }
  if (p.toggleOff) { dropIfIsolated(state.editSel); select(null); syncEditPanel(); redraw(); }
}

function changed(msg) {
  markEditDirty(); recordHistory(); syncEditPanel(); redraw();
  if (msg) status(msg, "ok");
}

// ── panel actions ─────────────────────────────────────────────────────────────
export function deselect() {
  dropIfIsolated(state.editSel);
  select(null); syncEditPanel(); redraw();
}

export function deleteSelectedNode() {
  if (state.busy || !state.editSel) return;
  const { kind, node, seg } = state.editSel;
  if (node < 0) {
    deleteSegment(kind, seg);
    select(null); changed("segment deleted");
    return;
  }
  const n = state.editable[kind];
  const nb = neighbours(n, node);
  const wasInlet = (n.inlet_nodes || []).includes(node);
  removeNode(kind, node);
  const m = state.editable[kind];
  if (m && wasInlet && !(m.inlet_nodes || []).length) {
    // keep the tree fed: hand the inlet to the deleted node's neighbour
    const heir = nb.length ? (nb[0] > node ? nb[0] - 1 : nb[0]) : null;
    if (heir != null && heir < m.node_x.length) { m.inlet_nodes = [heir]; m.root_node = heir; }
  }
  // keep drawing from where you were: select the (single) neighbour, if any
  if (m && nb.length === 1) select(kind, nb[0] > node ? nb[0] - 1 : nb[0]);
  else select(null);
  changed();
}

export function toggleInlet() {
  const sel = state.editSel;
  if (!sel || sel.node < 0) return;
  const n = state.editable[sel.kind];
  const inl = n.inlet_nodes || [];
  const word = sel.kind === "vein" ? "outlet" : "inlet";
  if (inl.includes(sel.node)) {
    if (inl.length === 1) { status(`the ${sel.kind} tree needs at least one ${word}`, "err"); return; }
    n.inlet_nodes = inl.filter((i) => i !== sel.node);
    if (n.root_node === sel.node) n.root_node = n.inlet_nodes[0];
  } else {
    n.inlet_nodes = [...inl, sel.node].sort((a, b) => a - b);
  }
  changed(`${word} ${inl.includes(sel.node) ? "removed" : "set"}`);
}

export function onRadiusInput() {
  const sel = state.editSel;
  if (!sel || sel.seg < 0) return;
  state.editable[sel.kind].seg_radius[sel.seg] = Math.max(1e-3, numOf("draw-radius"));
  $("radius-slider").value = numOf("draw-radius");
  markEditDirty(); syncEditPanel(); redraw();
}

export function onRadiusSlider() {
  $("draw-radius").value = +numOf("radius-slider").toFixed(2);
  onRadiusInput();
}

export function clearKind() {
  const kind = newKind();
  if (!state.editable[kind]) return;
  if (!confirm(`Delete the whole ${kind} tree?`)) return;
  state.editable[kind] = null;
  if (state.editSel && state.editSel.kind === kind) select(null);
  changed(`${kind} tree cleared`);
}

export function syncEditPanel() {
  const sel = state.editSel;
  const n = sel ? state.editable[sel.kind] : null;
  const del = $("btn-node-del"), io = $("btn-node-io"), split = $("btn-seg-split");
  const isSeg = !!(sel && n && sel.node < 0);
  del.disabled = !sel || state.busy;
  del.textContent = isSeg ? "🗑 Delete segment" : "🗑 Delete node";
  io.disabled = !sel || isSeg || state.busy;
  split.disabled = !isSeg || state.busy;
  $("radius-label").textContent = sel && sel.seg >= 0 ? "radius of selected segment" : "radius (new segments)";
  io.classList.remove("active");
  if (isSeg) {
    const a = n.seg_start[sel.seg], b = n.seg_end[sel.seg];
    const len = Math.hypot(n.node_x[b] - n.node_x[a], n.node_y[b] - n.node_y[a]);
    io.textContent = sel.kind === "vein" ? "▼ Set as outlet" : "▲ Set as inlet";
    $("node-info").innerHTML = `<b>${sel.kind}</b> segment #${sel.seg} · r = <b>${(+n.seg_radius[sel.seg]).toFixed(2)}</b>`
      + ` · length ${len.toFixed(1)}`;
  } else if (sel && n) {
    const isIn = (n.inlet_nodes || []).includes(sel.node);
    const word = sel.kind === "vein" ? "outlet" : "inlet";
    const tri = sel.kind === "vein" ? "▼" : "▲";
    io.textContent = isIn ? `${tri} Unset ${word}` : `${tri} Set as ${word}`;
    io.classList.toggle("active", isIn);
    const deg = degrees(n)[sel.node];
    const what = deg === 0 ? "new vessel start" : deg === 1 ? "end node" : deg === 2 ? "node" : "branch point";
    $("node-info").innerHTML = `<b>${sel.kind}</b> ${what} #${sel.node}`
      + (isIn ? ` · <b>${word}</b> (${sel.kind === "vein" ? "p_out" : "p_in"})` : "")
      + (sel.seg >= 0 ? ` · its segment r = <b>${(+n.seg_radius[sel.seg]).toFixed(2)}</b>` : "");
  } else {
    io.textContent = "▲▼ Set inlet/outlet";
    const counts = editKinds().map((k) => {
      const m = state.editable[k];
      return `${m.seg_start.length} ${k} segments, ${(m.inlet_nodes || []).length} ${k === "vein" ? "outlet" : "inlet"}(s)`;
    });
    $("node-info").textContent = counts.length ? counts.join(" · ")
      : "Empty — click the canvas to start a vessel (empty = default fractal trees on Run).";
  }
}

// ── persistence ───────────────────────────────────────────────────────────────
export function isEditDirty() { return editDirty; }
export function markEditDirty() { editDirty = true; $("btn-save-vessels").textContent = "Save vessels •"; }
export function clearEditDirty() { editDirty = false; $("btn-save-vessels").textContent = "Save vessels"; }

export async function saveEdits() {
  requireProject();
  dropIfIsolated(state.editSel);
  pruneIsolated("artery"); pruneIsolated("vein");
  const pack = (k) => {
    const n = state.editable[k];
    if (!n) return { node_x: [], node_y: [], seg_radius: [], seg_start: [], seg_end: [] };   // removes it
    return { node_x: n.node_x, node_y: n.node_y, seg_radius: n.seg_radius,
             seg_start: n.seg_start, seg_end: n.seg_end,
             inlet_nodes: n.inlet_nodes || null, root_node: n.root_node ?? null };
  };
  await api("PUT", `/projects/${state.project}/editable`, { artery: pack("artery"), vein: pack("vein") });
  if (state.legacyIO) {          // inlet/outlet points now live on the network nodes
    await api("PUT", `/projects/${state.project}/io`, { inlets: [], outlets: [] });
    state.legacyIO = false;
  }
  state.hasResult = false;                       // geometry changed -> result stale
  stopFlow(); state.tracer = null; state.fieldData = null;
  const keep = state.editSel;
  await Promise.all([loadNetwork(), loadEditable()]);
  if (keep && state.editable[keep.kind] && keep.node < state.editable[keep.kind].node_x.length) {
    select(keep.kind, keep.node);
  }
  syncEditPanel(); refreshUI(); redraw();
  status("vessels saved", "ok");
}

// Save pending edits before an action that reads the stored geometry (run, grow…).
export async function flushEdits() {
  if (editDirty && state.project) await saveEdits();
}

export async function revertEdits() {
  await loadEditable();
  syncEditPanel(); recordHistory(); redraw();
  status("reverted to saved vessels", "ok");
}
