// perfusion 2-D workspace — all canvas rendering: Darcy fields, tracer frames,
// the vessel-network overlay, edit handles, the direct-connection view and the
// drawing/tissue/IO overlays. Depends only on core.js + palette.js (no feature
// logic), so the rest of the app just calls redraw().
import {
  ctx, W, H, toCanvas, getCss, state, $, numOf, vtype, hasEditable, editKinds,
} from "./core.js";
import { VIRIDIS, viridis } from "./palette.js";

export function redraw() {
  ctx.clearRect(0, 0, W, H);
  if (state.mode === "result" && state.resultModel === "direct") {
    if (state.flow.active && state.tracer && state.tracer.network) {
      drawNetworkTracerFrame(state.flow.idx);   // bolus swept along the vessels
    } else {
      drawDirect();        // network + terminal connections coloured by flow (no field)
    }
    drawIO();
    return;
  }
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

// tracer frames are addressed by index, so the flow animation reaches in directly
export function drawTracerFrame(idx) {
  const t = state.tracer;
  paintHeatmap(t.frames[idx], t.shape[0], t.shape[1], 0, t.vmax);
  updateColorbar(0, t.vmax, "tracer");
}

// direct-model bolus animation: the vessel network as a faint substrate, with the
// contrast bolus lit up per element by its concentration at this frame (viridis).
function drawNetworkTracerFrame(idx) {
  const t = state.tracer;
  const conc = t.frames[idx];
  const artC = getCss("--art"), venC = getCss("--ven");
  ctx.lineCap = "round";
  t.lines.forEach((ln, i) => {
    const [ax, ay] = toCanvas(ln[0], ln[1]), [bx, by] = toCanvas(ln[2], ln[3]);
    const k = t.kind[i];
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.globalAlpha = k === 2 ? 0.1 : 0.18;      // faint anatomical substrate
    ctx.strokeStyle = k === 0 ? artC : k === 1 ? venC : "#10b981";
    ctx.lineWidth = k === 2 ? 0.7 : 1.2;
    ctx.stroke();
    const c = Math.max(0, Math.min(1, conc[i] / (t.vmax || 1)));
    if (c > 0.03) {                               // bolus overlay where present
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      ctx.globalAlpha = 0.25 + 0.75 * c;
      ctx.strokeStyle = viridis(c);
      ctx.lineWidth = (k === 2 ? 1.5 : 3) * (0.5 + c);
      ctx.stroke();
    }
  });
  ctx.globalAlpha = 1;
  updateColorbar(0, 1, "bolus");
}

function drawDirect() {
  const d = state.direct;
  if (!d) return;
  const fmax = d.flow_max || 1;
  ctx.globalAlpha = 0.22;                 // terminal-connection links (the capillary bed)
  ctx.strokeStyle = "#10b981"; ctx.lineWidth = 0.7;
  for (const [x0, y0, x1, y1] of d.connections) {
    const [ax, ay] = toCanvas(x0, y0), [bx, by] = toCanvas(x1, y1);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  const paint = (segs, color) => {
    for (const [x0, y0, x1, y1, f] of segs) {
      const [ax, ay] = toCanvas(x0, y0), [bx, by] = toCanvas(x1, y1);
      const t = Math.min(1, f / fmax);
      ctx.globalAlpha = 0.4 + 0.6 * t; ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, 5 * t); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
  };
  paint(d.artery, getCss("--art")); paint(d.vein, getCss("--ven"));
  ctx.globalAlpha = 1;
  $("colorbar").hidden = true;            // static direct view has no scalar scale
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
