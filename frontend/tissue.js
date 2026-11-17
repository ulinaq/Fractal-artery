// perfusion 2-D workspace — the tissue editor (Tissue tab).
//
// Three kinds of polygon share one editor — the organ outline (`regions` → mask),
// low-K lesions and growth hotspots — with the same node-editing feel as vessels:
//
//   click empty canvas  → start a new polygon of the "Draw" type; keep clicking
//   click the 1st point / Enter / double-click → close it
//   drag a vertex       → move it          click an outline → insert a vertex
//   drag inside         → move the polygon (and select it)
//   Del                 → delete the selected vertex (or the polygon if ≤3 left)
//   Esc                 → cancel the polygon in progress / deselect
//
// "Draw" picks the layer being edited (others are shown but not clickable).
// Selecting a lesion/hotspot loads its k / weight into the panel; editing the
// field updates that polygon.
import { api, numOf, state, status, toCanvas, toDomain, $ } from "./core.js";
import { redraw } from "./render.js";
import { recordHistory, requireProject } from "./app.js";

const VERT_TOL = 9;       // px
const EDGE_TOL = 6;       // px
const CLOSE_TOL = 10;     // px: click the first point to close
const DRAG_START = 4;
let dirty = false;
let press = null;         // {px, py, moved, kind: "vertex"|"poly", start:[x,y], orig:[[x,y]...]}

export const drawTarget = () => $("tissue-target").value;    // "organ" | "lesion" | "hotspot"

// Every polygon as {type, idx, poly} (poly is the live point array), topmost last.
function items() {
  return [
    ...state.regions.map((p, idx) => ({ type: "organ", idx, poly: p })),
    ...state.lesions.map((l, idx) => ({ type: "lesion", idx, poly: l.polygon })),
    ...state.hotspots.map((h, idx) => ({ type: "hotspot", idx, poly: h.polygon })),
  ];
}
const listOf = (type) => (type === "lesion" ? state.lesions : type === "hotspot" ? state.hotspots : state.regions);
const polyOf = (type, idx) => (type === "organ" ? state.regions[idx] : listOf(type)[idx].polygon);

function pointInPoly([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  return { d: (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2, t };
}

// Only the layer chosen in "Draw" is editable (the others stay visible), so a
// lesion or hotspot can be drawn *inside* the organ outline.
function hitTest(px, py) {
  const layer = drawTarget();
  const all = items().filter((it) => it.type === layer).reverse();   // topmost first
  for (const it of all) {
    for (let v = 0; v < it.poly.length; v++) {
      const [cx, cy] = toCanvas(...it.poly[v]);
      if ((cx - px) ** 2 + (cy - py) ** 2 < VERT_TOL ** 2) return { ...it, vertex: v };
    }
  }
  for (const it of all) {
    for (let v = 0; v < it.poly.length; v++) {
      const a = toCanvas(...it.poly[v]), b = toCanvas(...it.poly[(v + 1) % it.poly.length]);
      const { d, t } = segDist2(px, py, a[0], a[1], b[0], b[1]);
      if (d < EDGE_TOL ** 2) return { ...it, edge: v, t };
    }
  }
  const pt = toDomain(px, py);
  for (const it of all) if (pointInPoly(pt, it.poly)) return { ...it, inside: true };
  return null;
}

function select(type, idx, vertex = null) {
  state.tissueSel = type == null ? null : { type, idx, vertex };
  if (type != null) {
    $("tissue-target").value = type;
    syncTargetRows();
    if (type === "lesion") $("lesion-k").value = state.lesions[idx].permeability;
    if (type === "hotspot") $("hotspot-w").value = state.hotspots[idx].weight;
  }
  syncTissuePanel();
}

// ── pointer interaction ───────────────────────────────────────────────────────
export function tissuePointerDown(px, py) {
  if (!state.project) { status("create a project first", "err"); return; }
  const pt = toDomain(px, py);
  const stroke = state.tissueStroke;
  if (stroke.length) {                                     // a polygon is in progress
    const [fx, fy] = toCanvas(...stroke[0]);
    if (stroke.length >= 3 && (fx - px) ** 2 + (fy - py) ** 2 < CLOSE_TOL ** 2) closePolygon();
    else { stroke.push(pt); redraw(); }
    return;
  }
  const hit = hitTest(px, py);
  if (!hit) {
    if (state.tissueSel) { select(null); redraw(); return; }   // first click just deselects
    state.tissueStroke = [pt];
    status(`drawing ${drawTarget()}: click to add points · click the first point or Enter to close`);
    redraw();
    return;
  }
  const poly = polyOf(hit.type, hit.idx);
  if (hit.edge != null) {                                  // insert a vertex on the outline
    poly.splice(hit.edge + 1, 0, pt);
    select(hit.type, hit.idx, hit.edge + 1);
    press = { px, py, moved: true, kind: "vertex" };      // already a change
    markDirty(); redraw();
    return;
  }
  if (hit.vertex != null) {
    select(hit.type, hit.idx, hit.vertex);
    press = { px, py, moved: false, kind: "vertex" };
  } else {
    select(hit.type, hit.idx, null);
    press = { px, py, moved: false, kind: "poly", start: pt, orig: poly.map((p) => [...p]) };
  }
  redraw();
}

export function tissuePointerMove(px, py) {
  if (!press) {
    if (state.tissueStroke.length) { state.hover = toDomain(px, py); redraw(); }
    return;
  }
  if (!press.moved && Math.hypot(px - press.px, py - press.py) < DRAG_START) return;
  press.moved = true;
  const sel = state.tissueSel, pt = toDomain(px, py);
  const poly = polyOf(sel.type, sel.idx);
  if (press.kind === "vertex") {
    poly[sel.vertex] = pt;
  } else {
    const dx = pt[0] - press.start[0], dy = pt[1] - press.start[1];
    press.orig.forEach(([x, y], i) => { poly[i] = [x + dx, y + dy]; });
  }
  redraw();
}

export function tissuePointerUp() {
  if (!press) return;
  const moved = press.moved; press = null;
  if (moved) { markDirty(); recordHistory(); }
}

// ── actions ───────────────────────────────────────────────────────────────────
export function closePolygon() {
  const s = state.tissueStroke;
  // a double-click lands two extra clicks on the same spot — drop near-duplicates
  while (s.length >= 2) {
    const [a, b] = [toCanvas(...s.at(-1)), toCanvas(...s.at(-2))];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 4) s.pop(); else break;
  }
  state.tissueStroke = [];
  if (s.length < 3) { status("a region needs at least 3 points", "err"); redraw(); return; }
  const t = drawTarget(), poly = s.map(([x, y]) => [x, y]);
  if (t === "lesion") state.lesions.push({ polygon: poly, permeability: Math.max(1e-3, numOf("lesion-k")) });
  else if (t === "hotspot") state.hotspots.push({ polygon: poly, weight: Math.max(1, numOf("hotspot-w")) });
  else state.regions.push(poly);
  select(t, listOf(t).length - 1);
  markDirty(); recordHistory(); redraw();
  status(`${t} region added`, "ok");
}

export function cancelOrDeselect() {
  if (state.tissueStroke.length) state.tissueStroke = [];
  else select(null);
  redraw();
}

export function deleteSelection() {
  if (state.tissueStroke.length) { state.tissueStroke.pop(); redraw(); return; }
  const sel = state.tissueSel;
  if (!sel) return;
  const poly = polyOf(sel.type, sel.idx);
  if (sel.vertex != null && poly.length > 3) {
    poly.splice(sel.vertex, 1);
    select(sel.type, sel.idx, null);
  } else {
    listOf(sel.type).splice(sel.idx, 1);
    select(null);
  }
  markDirty(); recordHistory(); redraw();
}

export function clearTarget() {
  const t = drawTarget(), n = listOf(t).length;
  if (!n) return;
  if (!confirm(`Delete all ${n} ${t} region(s)?`)) return;
  if (t === "lesion") state.lesions = [];
  else if (t === "hotspot") state.hotspots = [];
  else state.regions = [];
  select(null); markDirty(); recordHistory(); redraw();
}

// k / weight inputs edit the selected lesion / hotspot
export function onTissueValue() {
  const sel = state.tissueSel;
  if (!sel) return;
  if (sel.type === "lesion") state.lesions[sel.idx].permeability = Math.max(1e-3, numOf("lesion-k"));
  else if (sel.type === "hotspot") state.hotspots[sel.idx].weight = Math.max(1, numOf("hotspot-w"));
  else return;
  markDirty(); redraw();
}

export function syncTargetRows() {
  $("lesion-k-row").style.display = drawTarget() === "lesion" ? "" : "none";
  $("hotspot-w-row").style.display = drawTarget() === "hotspot" ? "" : "none";
}

export function syncTissuePanel() {
  const sel = state.tissueSel;
  $("btn-tissue-del").disabled = !sel || state.busy;
  if (sel) {
    const poly = polyOf(sel.type, sel.idx);
    $("tissue-info").innerHTML = `<b>${sel.type}</b> region #${sel.idx + 1} · ${poly.length} points`
      + (sel.vertex != null ? ` · vertex ${sel.vertex + 1} selected` : "");
    $("btn-tissue-del").textContent = sel.vertex != null && poly.length > 3 ? "🗑 Delete point" : "🗑 Delete region";
  } else {
    $("btn-tissue-del").textContent = "🗑 Delete";
    const bits = [`${state.regions.length} organ`, `${state.lesions.length} lesion`, `${state.hotspots.length} hotspot`];
    $("tissue-info").textContent = `${bits.join(" · ")} region(s)`
      + (state.regions.length ? "" : " — no organ outline = whole domain is tissue.");
  }
}

// ── persistence ───────────────────────────────────────────────────────────────
export function markDirty() { dirty = true; $("btn-save-mask").textContent = "Save tissue •"; syncTissuePanel(); }
export function clearTissueDirty() { dirty = false; $("btn-save-mask").textContent = "Save tissue"; }

export async function saveMask() {
  requireProject();
  if (state.tissueStroke.length >= 3) closePolygon();
  await api("PUT", `/projects/${state.project}/mask`, { regions: state.regions });
  await api("PUT", `/projects/${state.project}/lesions`, { lesions: state.lesions });
  await api("PUT", `/projects/${state.project}/hotspots`, { hotspots: state.hotspots });
  clearTissueDirty();
  const bits = [state.regions.length ? `${state.regions.length} region(s)` : "full domain"];
  if (state.lesions.length) bits.push(`${state.lesions.length} lesion(s)`);
  if (state.hotspots.length) bits.push(`${state.hotspots.length} hotspot(s)`);
  status(`saved tissue — ${bits.join(", ")}`, "ok");
}

export async function flushTissue() {
  if ((dirty || state.tissueStroke.length >= 3) && state.project) await saveMask();
}

export async function loadMask() {
  const get = async (path, key) => {
    try { return (await api("GET", `/projects/${state.project}/${path}`))[key] || []; }
    catch { return []; }
  };
  state.regions = await get("mask", "regions");
  state.lesions = await get("lesions", "lesions");
  state.hotspots = await get("hotspots", "hotspots");
  state.tissueStroke = []; state.tissueSel = null;
  clearTissueDirty(); syncTissuePanel();
}
