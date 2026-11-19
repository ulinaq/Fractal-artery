// perfusion 2-D workspace — shared foundation. A leaf module (imports no feature
// code) holding the domain/canvas geometry, the API client, the mutable app
// state and small DOM helpers, so render.js and app.js both build on it.

// ── domain / canvas geometry ────────────────────────────────────────────────
// The canvas fills the stage and zooms/pans over the physical window (see
// viewport.js). W/H are its logical (CSS-pixel) size — live bindings, updated
// on resize — and toCanvas/toDomain include the current zoom + pan.
import { fetchJSON } from "./net.js";
import { createViewport } from "./viewport.js";

// the physical window — per project (Parameters → domain size; restored on open)
export const DOMAIN = { xmin: 0, xmax: 100, ymin: 0, ymax: 120 };
export const GRID_L = { lx: DOMAIN.xmax - DOMAIN.xmin, ly: DOMAIN.ymax - DOMAIN.ymin };
export const canvas = document.getElementById("view");
export const ctx = canvas.getContext("2d");
export let W = canvas.width, H = canvas.height;
const viewListeners = [];
export const onViewChange = (fn) => viewListeners.push(fn);
export const vp = createViewport(canvas, {
  domain: () => [DOMAIN.xmin, DOMAIN.xmax, DOMAIN.ymin, DOMAIN.ymax],
  // in Result there is nothing to edit, so a plain drag pans the view
  panWithLeft: () => state.mode === "result",
  onResize: (v) => { W = v.W; H = v.H; },
  onChange: () => viewListeners.forEach((fn) => fn()),
});
export const toCanvas = (x, y) => vp.toCanvas(x, y);
/** switch the physical window (a bigger/smaller object) and refit the view */
export function setDomain(xmin, xmax, ymin, ymax) {
  Object.assign(DOMAIN, { xmin, xmax, ymin, ymax });
  Object.assign(GRID_L, { lx: xmax - xmin, ly: ymax - ymin });
  vp.fit();
}
export const toDomain = (px, py) => vp.toDomain(px, py);
/** canvas pixels per domain unit relative to the original fixed 640-px canvas —
 *  scales radius-based line widths so vessels thicken as you zoom in */
export const zoomPx = () => vp.scale / 6.4;

// ── tiny API client ───────────────────────────────────────────────────────────
// throws an Error carrying the server's (readable) message and .status
export const api = (method, path, body) => fetchJSON(method, path, body);

// ── application state ───────────────────────────────────────────────────────
export const state = {
  project: null,
  mode: "vessels",           // "vessels" | "tissue" | "result"
  field: "pressure",         // the base result field shown / reloaded (pressure|flux|tof)
  fieldLabel: "pressure",    // what the colour bar calls the map on screen (Δ, O₂, Fp, …)
  vessels: { arteries: [], veins: [] },   // legacy hand-drawn polylines (vectorized on open)
  regions: [],               // tissue outline polygons (domain coords) -> mask
  lesions: [],               // low-K patches: {polygon:[[x,y]...], permeability}
  hotspots: [],              // growth-demand targets: {polygon:[[x,y]...], weight}
  tissueStroke: [],          // polygon being drawn in the Tissue tab
  tissueSel: null,           // {type, idx, vertex} selected tissue polygon / vertex
  inlets: [],                // legacy inlet points (adopted onto network nodes on open)
  outlets: [],               // legacy outlet points
  legacyIO: false,           // legacy points were adopted -> clear them server-side on save
  network: null,             // {segments:[...]} overlaid in result mode
  editable: { artery: null, vein: null },  // the vessel networks being edited
  editSel: null,             // {kind, node, seg} selected vessel node
  hover: null,               // pointer position (domain) for rubber-band previews
  fieldData: null,           // {name, shape, data, vmin, vmax}
  tracer: null,              // {shape, times, vmax, frames:[...]}
  flow: { active: false, idx: 0, timer: null },
  busy: false,               // a job is running -> lock actions
  hasResult: false,          // a solved result exists (enables Export)
  resultModel: "hybrid",     // "hybrid" (field) | "direct" (network + connections)
  direct: null,              // direct-connection render data {artery,vein,connections,flow_max}
  runNotes: [],              // solver warnings (e.g. non-convergence)
};

// ── DOM helpers ───────────────────────────────────────────────────────────────
export const $ = (id) => document.getElementById(id);
const statusEl = $("status");
export function status(text, kind = "") { statusEl.textContent = text; statusEl.className = "status " + kind; }
export const numOf = (id) => Number($(id).value);
export const vtype = () => document.querySelector('input[name=vtype]:checked').value;
export const getCss = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// editable-network state readers (used by both the edit logic and the renderer)
export function hasEditable() { return !!(state.editable && (state.editable.artery || state.editable.vein)); }
export function editKinds() { return ["artery", "vein"].filter((k) => state.editable && state.editable[k]); }
