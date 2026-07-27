// perfusion — tree explorer: interactively tune a fractal tree and see the
// perfusion field, vessel geometry and homogeneity objective update live, with a
// one-click optimiser (Paper D). Self-contained, talks to /explore.
import { VIRIDIS, viridis } from "./palette.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const PARAMS = ["a_gamma", "v_gamma", "length_to_radius", "exponent", "rv_scale"];

const status = (t, k = "") => { $("status").textContent = t; $("status").className = "status " + k; };
const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "content-type": "application/json" },
                                body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 150)}`);
  return r.json();
}

function params() {
  const p = { kind: $("kind").value, field: $("field").value, nx: 60, ny: 70 };
  for (const k of PARAMS) p[k] = Number($(k).value);
  return p;
}

function syncLabels() {
  for (const k of PARAMS) $("v-" + k).textContent = Number($(k).value).toFixed(2);
}

let busy = false, queued = false;
async function evaluate() {
  if (busy) { queued = true; return; }        // coalesce rapid slider moves
  busy = true;
  syncLabels();
  status("solving…");
  try {
    const d = await post("/explore", params());
    draw(d);
    $("obj-info").textContent = `${d.kind} = ${fmt(d.objective)}`;
    status("ready", "ok");
  } catch (e) {
    status(e.message, "err");
  } finally {
    busy = false;
    if (queued) { queued = false; evaluate(); }
  }
}

function draw(d) {
  ctx.clearRect(0, 0, W, H);
  const [x0, x1, y0, y1] = d.domain;
  const tc = (x, y) => [(x - x0) / (x1 - x0) * W, H - (y - y0) / (y1 - y0) * H];
  const { data, shape, vmin, vmax } = d.field;
  paintHeatmap(data, shape[0], shape[1], vmin, vmax);
  for (const [ax, ay, bx, by, r, kind] of d.segments) {
    const [p0x, p0y] = tc(ax, ay), [p1x, p1y] = tc(bx, by);
    ctx.strokeStyle = kind === 1 ? "rgba(37,99,235,.9)" : "rgba(220,38,38,.9)";
    ctx.lineWidth = Math.max(0.5, r * 0.4);
    ctx.beginPath(); ctx.moveTo(p0x, p0y); ctx.lineTo(p1x, p1y); ctx.stroke();
  }
  updateColorbar(vmin, vmax, d.field.name);
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
      const o = ((ny - 1 - iy) * nx + ix) * 4;
      if (v === null) { img.data[o + 3] = 0; continue; }
      const [r, g, b] = viridis((v - vmin) / span);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
}

function updateColorbar(vmin, vmax, label) {
  $("colorbar").hidden = false;
  const stops = VIRIDIS.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i / (VIRIDIS.length - 1)).toFixed(0)}%`);
  $("cb-ramp").style.background = `linear-gradient(to top, ${stops.join(",")})`;
  $("cb-max").textContent = fmt(vmax); $("cb-min").textContent = fmt(vmin);
  $("cb-title").textContent = label;
}

async function optimise() {
  $("btn-opt").disabled = true;
  status("optimising… (this runs several solves)");
  try {
    const o = await post("/explore/optimize", {
      kind: $("kind").value, method: $("method").value, maxiter: Number($("maxiter").value),
      nx: 50, ny: 58,
    });
    for (const k of PARAMS) if (o[k] != null) $(k).value = o[k];   // load the best config
    syncLabels();
    $("opt-info").textContent =
      `${o.kind}: ${fmt(o.start)} → ${fmt(o.best)} (${(o.improvement * 100).toFixed(1)}% better, ${o.n_eval} evals)`;
    await evaluate();
  } catch (e) {
    status(e.message, "err");
  } finally {
    $("btn-opt").disabled = false;
  }
}

// ── interaction ────────────────────────────────────────────────────────────
let debounce = null;
for (const k of PARAMS) {
  $(k).addEventListener("input", () => {
    syncLabels();
    clearTimeout(debounce);
    debounce = setTimeout(evaluate, 250);
  });
}
$("kind").addEventListener("change", evaluate);
$("field").addEventListener("change", evaluate);
$("btn-opt").addEventListener("click", optimise);

evaluate();
