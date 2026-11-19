// perfusion 2-D workspace — all canvas rendering: Darcy fields, tracer frames,
// the vessel-network overlay, edit handles, the direct-connection view and the
// drawing/tissue/IO overlays. Depends only on core.js + palette.js (no feature
// logic), so the rest of the app just calls redraw().
import { ctx, DOMAIN, W, H, toCanvas, getCss, state, $, editKinds, vp, zoomPx } from "./core.js";
import { DIVERGING, VIRIDIS, diverging, viridis } from "./palette.js";

export function redraw() {
  vp.begin();                    // HiDPI transform + clear
  drawBackdrop();
  if (state.mode !== "result") $("colorbar").hidden = true;   // fields only show in Result
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
  if (state.mode === "result") drawNetworkOverlay();
  else drawEditNetwork();
  drawTissueStroke();
  drawIO();
}

// the physical domain as a white sheet on a soft backdrop, so zoom/pan reads clearly
function drawBackdrop() {
  ctx.fillStyle = "#f1f5f9"; ctx.fillRect(0, 0, W, H);
  const [x0, y0] = toCanvas(DOMAIN.xmin, DOMAIN.ymax), [x1, y1] = toCanvas(DOMAIN.xmax, DOMAIN.ymin);
  ctx.fillStyle = "#fff"; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.strokeStyle = "#cbd5e1"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
}

// radius (domain units) → on-screen line width, thickening as you zoom in
const rw = (r, k = 1, min = 1) => Math.max(min, r * k * zoomPx());

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
      ctx.lineWidth = Math.max(1, 5 * t * Math.sqrt(zoomPx())); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
  };
  paint(d.artery, getCss("--art")); paint(d.vein, getCss("--ven"));
  ctx.globalAlpha = 1;
  $("colorbar").hidden = true;            // static direct view has no scalar scale
}

// inlets ▲ (artery, p_in) and outlets ▼ (vein, p_out) are the networks'
// prescribed-pressure nodes (`inlet_nodes`), drawn in every view.
function drawIO() {
  const marker = (x, y, up, color) => {
    const [px, py] = toCanvas(x, y), s = 7;
    ctx.beginPath();
    if (up) { ctx.moveTo(px, py - s - 2); ctx.lineTo(px - s, py + s - 2); ctx.lineTo(px + s, py + s - 2); }
    else { ctx.moveTo(px, py + s + 2); ctx.lineTo(px - s, py - s + 2); ctx.lineTo(px + s, py - s + 2); }
    ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
  };
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    for (const i of n.inlet_nodes || []) {
      if (i < n.node_x.length) marker(n.node_x[i], n.node_y[i], kind === "artery", getCss(kind === "vein" ? "--ven" : "--art"));
    }
  }
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
  drawLesions();
  drawHotspots();
  drawTissueHandles();
}

// Tissue tab: vertex handles on every polygon, the selected one emphasised.
function drawTissueHandles() {
  if (state.mode !== "tissue") return;
  const sel = state.tissueSel, layer = $("tissue-target").value;
  const groups = [["organ", state.regions.map((p) => p), "#047857"],
                  ["lesion", state.lesions.map((l) => l.polygon), "#b45309"],
                  ["hotspot", state.hotspots.map((h) => h.polygon), "#0f766e"]];
  for (const [type, polys, color] of groups) {
    if (type !== layer) continue;                 // handles only on the editable layer
    polys.forEach((poly, idx) => {
      const isSel = sel && sel.type === type && sel.idx === idx;
      if (isSel) {
        ctx.beginPath(); tracePath(poly); ctx.closePath();
        ctx.strokeStyle = getCss("--accent"); ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]);
      }
      poly.forEach(([x, y], v) => {
        const [px, py] = toCanvas(x, y);
        const vSel = isSel && sel.vertex === v, s = vSel ? 5 : isSel ? 4 : 3;
        ctx.fillStyle = vSel ? getCss("--accent") : "#fff";
        ctx.strokeStyle = isSel ? getCss("--accent") : color; ctx.lineWidth = 1.5;
        ctx.fillRect(px - s, py - s, 2 * s, 2 * s); ctx.strokeRect(px - s, py - s, 2 * s, 2 * s);
      });
    });
  }
}

// growth-demand hotspots: teal outline + fill, opacity growing with the demand weight;
// shown in tissue mode (to edit) and over the result.
function drawHotspots() {
  if (state.mode !== "tissue" && state.mode !== "result") return;
  for (const h of state.hotspots) {
    const poly = h.polygon || h;
    const w = h.weight ?? 4;
    ctx.beginPath(); tracePath(poly); ctx.closePath();
    ctx.fillStyle = `rgba(13,148,136,${Math.min(0.4, 0.08 + 0.05 * w).toFixed(3)})`;
    ctx.fill();
    ctx.strokeStyle = "rgba(15,118,110,.9)"; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([]);
    if (poly.length) {
      const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length;
      const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
      const [tx, ty] = toCanvas(cx, cy);
      ctx.fillStyle = "#134e4a"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
      ctx.fillText(`×${+w}`, tx, ty);
      ctx.textAlign = "start";
    }
  }
}

// low-K permeability lesions: amber outline, opacity growing as K falls; shown in
// tissue mode (to edit) and over the result (to read alongside the field).
function drawLesions() {
  if (state.mode !== "tissue" && state.mode !== "result") return;
  for (const les of state.lesions) {
    const poly = les.polygon || les;
    const k = les.permeability ?? 1;
    ctx.beginPath(); tracePath(poly); ctx.closePath();
    // over a Δ map only the outline: the change inside the lesion is the point
    if (!(state.mode === "result" && state.fieldData && state.fieldData.delta)) {
      ctx.fillStyle = `rgba(217,119,6,${(0.15 + 0.4 * (1 - Math.min(1, k))).toFixed(3)})`;
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(180,83,9,.9)"; ctx.lineWidth = 1.5; ctx.stroke();
    if (poly.length) {
      const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length;
      const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
      const [tx, ty] = toCanvas(cx, cy);
      ctx.fillStyle = "#7c2d12"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
      ctx.fillText(`k=${(+k).toPrecision(2)}`, tx, ty);
      ctx.textAlign = "start";
    }
  }
}

function paintHeatmap(data, nx, ny, vmin, vmax, ramp = viridis) {
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
      const [r, g, b] = ramp((v - vmin) / span);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const [x0, y0] = toCanvas(DOMAIN.xmin, DOMAIN.ymax), [x1, y1] = toCanvas(DOMAIN.xmax, DOMAIN.ymin);
  ctx.imageSmoothingEnabled = vp.zoom < 3;            // zoomed in: show the grid cells
  ctx.drawImage(off, x0, y0, x1 - x0, y1 - y0);
}

function drawField() {
  const { shape, data, vmin, vmax, delta } = state.fieldData;
  paintHeatmap(data, shape[0], shape[1], vmin, vmax, delta ? diverging : viridis);
  updateColorbar(vmin, vmax, state.fieldLabel || state.field, delta ? DIVERGING : VIRIDIS);
}

// Vessels tab: the editable networks with node handles, the selection and a
// rubber-band preview of the next edge. Tissue tab: the same vessels, faint.
// The default fractal tree a run would use for any kind you haven't drawn: faint
// and dashed, so it's clear what gets solved (it vanishes once you draw that kind).
function drawDefaultTrees() {
  if (!state.network) return;
  const segs = state.network.segments.filter((sg) => sg.default && !state.editable[sg.kind]);
  if (!segs.length) return;
  ctx.save();
  ctx.globalAlpha = 0.35; ctx.setLineDash([4, 4]); ctx.lineCap = "round";
  for (const sg of segs) {
    const [x0, y0] = toCanvas(sg.x0, sg.y0), [x1, y1] = toCanvas(sg.x1, sg.y1);
    ctx.strokeStyle = getCss(sg.kind === "vein" ? "--ven" : "--art");
    ctx.lineWidth = rw(sg.r, 0.6, 0.8);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  ctx.restore();
}

function drawEditNetwork() {
  drawDefaultTrees();
  const editing = state.mode === "vessels";
  const sel = state.editSel;
  for (const kind of editKinds()) {
    const n = state.editable[kind];
    const color = getCss(kind === "vein" ? "--ven" : "--art");
    ctx.globalAlpha = editing ? 0.75 : 0.3; ctx.lineCap = "round";
    ctx.strokeStyle = color;
    for (let s = 0; s < n.seg_start.length; s++) {
      const [x0, y0] = toCanvas(n.node_x[n.seg_start[s]], n.node_y[n.seg_start[s]]);
      const [x1, y1] = toCanvas(n.node_x[n.seg_end[s]], n.node_y[n.seg_end[s]]);
      ctx.lineWidth = rw(n.seg_radius[s], 1, 1.2);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (!editing) continue;
    if (sel && sel.kind === kind && sel.seg >= 0 && sel.seg < n.seg_start.length) {
      // the selected segment (or the selected node's edge): a halo at its true width
      const s = sel.seg, segSel = sel.node < 0;
      const [x0, y0] = toCanvas(n.node_x[n.seg_start[s]], n.node_y[n.seg_start[s]]);
      const [x1, y1] = toCanvas(n.node_x[n.seg_end[s]], n.node_y[n.seg_end[s]]);
      ctx.strokeStyle = getCss("--accent");
      if (segSel) {
        ctx.globalAlpha = 0.25; ctx.lineWidth = rw(n.seg_radius[s], 1, 1.2) + 10;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        ctx.globalAlpha = 1; ctx.lineWidth = rw(n.seg_radius[s], 1, 2);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      } else {
        ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    const dense = n.node_x.length > 400;
    for (let i = 0; i < n.node_x.length; i++) {
      const isSel = sel && sel.kind === kind && sel.node === i;
      const [cx, cy] = toCanvas(n.node_x[i], n.node_y[i]);
      ctx.beginPath(); ctx.arc(cx, cy, isSel ? 6 : dense ? 2 : 3, 0, 7);
      ctx.fillStyle = isSel ? getCss("--accent") : "#fff";
      ctx.fill(); ctx.lineWidth = isSel ? 2 : 1.3; ctx.strokeStyle = isSel ? "#fff" : color; ctx.stroke();
      if (isSel) { ctx.beginPath(); ctx.arc(cx, cy, 9, 0, 7); ctx.strokeStyle = getCss("--accent"); ctx.lineWidth = 1.5; ctx.stroke(); }
    }
  }
  if (editing && sel && sel.node >= 0 && state.hover && state.editable[sel.kind]) {   // where the next click lands
    const n = state.editable[sel.kind];
    const [ax, ay] = toCanvas(n.node_x[sel.node], n.node_y[sel.node]);
    const [bx, by] = toCanvas(...state.hover);
    ctx.strokeStyle = getCss(sel.kind === "vein" ? "--ven" : "--art");
    ctx.globalAlpha = 0.45; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
}

function drawNetworkOverlay() {
  if (!state.network) return;          // result view: the saved (solved) geometry
  for (const s of state.network.segments) {
    const [x0, y0] = toCanvas(s.x0, s.y0), [x1, y1] = toCanvas(s.x1, s.y1);
    ctx.strokeStyle = s.kind === "vein" ? "rgba(191,219,254,.95)" : "rgba(254,202,202,.95)";
    ctx.lineWidth = rw(s.r, 0.5, 0.6);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
}

// the tissue polygon being drawn, with a rubber band to the pointer and a ring
// on the first point (click it to close)
function drawTissueStroke() {
  const pts = state.tissueStroke;
  if (state.mode !== "tissue" || !pts.length) return;
  const t = $("tissue-target").value;
  const color = t === "lesion" ? "rgba(180,83,9,.95)" : t === "hotspot" ? "rgba(15,118,110,.95)" : "rgba(5,150,105,.95)";
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.8;
  ctx.beginPath(); tracePath(pts);
  if (state.hover) { const [hx, hy] = toCanvas(...state.hover); ctx.lineTo(hx, hy); }
  ctx.stroke();
  for (const [x, y] of pts) { const [px, py] = toCanvas(x, y); ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill(); }
  if (pts.length >= 3) {
    const [fx, fy] = toCanvas(...pts[0]);
    ctx.beginPath(); ctx.arc(fx, fy, 8, 0, 7); ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function updateColorbar(vmin, vmax, label, ramp = VIRIDIS) {
  const cb = $("colorbar"); cb.hidden = false;
  const stops = ramp.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${(100 * i / (ramp.length - 1)).toFixed(0)}%`);
  $("cb-ramp").style.background = `linear-gradient(to top, ${stops.join(",")})`;
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(1) : v.toFixed(2);
  $("cb-max").textContent = fmt(vmax); $("cb-min").textContent = fmt(vmin);
  $("cb-title").textContent = label;
}
