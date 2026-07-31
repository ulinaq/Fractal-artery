// perfusion 2-D workspace — shared foundation. A leaf module (imports no feature
// code) holding the domain/canvas geometry, the API client, the mutable app
// state and small DOM helpers, so render.js and app.js both build on it.

// ── domain / canvas geometry ────────────────────────────────────────────────
export const DOMAIN = { xmin: 0, xmax: 100, ymin: 0, ymax: 120 };   // physical window
export const GRID_L = { lx: DOMAIN.xmax - DOMAIN.xmin, ly: DOMAIN.ymax - DOMAIN.ymin };
export const canvas = document.getElementById("view");
export const ctx = canvas.getContext("2d");
export const W = canvas.width, H = canvas.height;
const sx = W / (DOMAIN.xmax - DOMAIN.xmin);
const sy = H / (DOMAIN.ymax - DOMAIN.ymin);
export const toCanvas = (x, y) => [(x - DOMAIN.xmin) * sx, H - (y - DOMAIN.ymin) * sy];
export const toDomain = (px, py) => [DOMAIN.xmin + px / sx, DOMAIN.ymin + (H - py) / sy];

// ── tiny API client ───────────────────────────────────────────────────────────
export async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["content-type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.status === 204 ? null : r.json();
}

// ── application state ───────────────────────────────────────────────────────
export const state = {
  project: null,
  mode: "vessels",           // "vessels" | "tissue" | "io" | "edit" | "result"
  field: "pressure",
  vessels: { arteries: [], veins: [] },   // saved strokes
  stroke: [],                // current in-progress vessel polyline (domain coords)
  regions: [],               // saved tissue outline polygons (domain coords)
  regionStroke: [],          // current in-progress tissue polygon
  lesions: [],               // low-K patches: {polygon:[[x,y]...], permeability}
  lesionStroke: [],          // current in-progress lesion polygon
  hotspots: [],              // growth-demand targets: {polygon:[[x,y]...], weight}
  hotspotStroke: [],         // current in-progress hotspot polygon
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
