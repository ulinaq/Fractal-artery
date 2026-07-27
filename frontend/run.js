// perfusion 2-D workspace — grow / solve / results / tracer.
// Kicks off grow & run jobs, loads the result view (Darcy field or the
// direct-connection network), and drives the tracer flow animation.
//
// Cross-module refs (app wires the buttons; edit reuses loadNetwork/stopFlow) are
// resolved by ES-module live bindings — every call happens at runtime.
import { $, api, getCss, numOf, state, status } from "./core.js";
import { redraw } from "./render.js";
import { loadEditable } from "./edit.js";
import {
  recordHistory, requireProject, saveIO, saveMask, saveVessels, setMode,
} from "./app.js";

export async function grow() {
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

export async function run() {
  requireProject();
  if (state.vessels.arteries.length || state.vessels.veins.length) await saveVessels();
  if (state.regions.length || state.regionStroke.length >= 3) await saveMask();
  if (state.inlets.length || state.outlets.length) await saveIO();
  const job = await api("POST", `/projects/${state.project}/run`,
    { model: $("run-model").value, nonlinear: $("nonlinear").checked,
      use_vessels: $("use-vessels").checked, max_iter: numOf("max-iter"),
      rho_c: numOf("rho-c"), phase_separation: $("phase-sep").checked,
      elasticity: $("elasticity").checked, compliance: numOf("compliance") });
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
  await loadNetwork();
  await loadResultView();
  setMode("result");
  if (state.runNotes && state.runNotes.length) {   // e.g. non-convergence within max iterations
    status("solved (with warnings)", "ok");
    $("run-info").textContent = "⚠ " + state.runNotes.join(" · ") +
      " — increase max iterations if the result looks off.";
  }
}

// Load the result view for whichever model solved: a Darcy field, or the
// direct-connection network (vessels + terminal links).
export async function loadResultView() {
  await showSummary();                       // sets state.resultModel + runNotes
  state.direct = null;
  if (state.resultModel === "direct") await loadDirect();
  else await loadField(state.field);
  await loadTransit();                        // h(τ) chart (hybrid) or a note (direct)
}

async function loadDirect() {
  try { state.direct = await api("GET", `/projects/${state.project}/direct`); }
  catch { state.direct = null; }
}

// ── transit-time distribution chart ────────────────────────────────────────────
const ttdFmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(1) : v.toFixed(2);

async function loadTransit() {
  const el = $("ttd");
  if (!el) return;
  const ctx = el.getContext("2d");
  ctx.clearRect(0, 0, el.width, el.height);
  if (state.resultModel !== "hybrid") {
    $("ttd-info").textContent = "the direct-connection model has no transit-time field";
    return;
  }
  let d;
  try { d = await api("GET", `/projects/${state.project}/transit?bins=32`); }
  catch { $("ttd-info").textContent = ""; return; }
  drawTTD(el, d);
  $("ttd-info").textContent =
    `MTT ${ttdFmt(d.mtt)} · CTH ${ttdFmt(d.cth)} · RD ${d.rd.toFixed(2)} · OEF ${(d.oef * 100).toFixed(0)}%`;
}

// ── tracer-kinetic model fit ────────────────────────────────────────────────
export async function fitKinetics() {
  requireProject();
  if (state.resultModel !== "hybrid") { status("kinetics needs the hybrid tracer", "err"); return; }
  let d;
  try {
    d = await api("GET", `/projects/${state.project}/kinetics?model=${$("tkm-model").value}`);
  } catch (e) {
    status(/409/.test(e.message) ? "compute the ▶ Tracer first" : e.message, "err");
    return;
  }
  drawTKM($("tkm-chart"), d);
  const parts = [`Fp=${ttdFmt(d.Fp)}`];
  if (d.vp != null) parts.push(`vp=${ttdFmt(d.vp)}`);
  if (d.PS != null) parts.push(`PS=${ttdFmt(d.PS)}`);
  if (d.ve != null) parts.push(`ve=${ttdFmt(d.ve)}`);
  $("tkm-info").textContent = `${d.model}: ${parts.join(" · ")}`;
  status("kinetics fitted", "ok");
}

// ── transient (pulsatile / Windkessel) response ─────────────────────────────
export async function runTransient() {
  requireProject();
  if (state.resultModel !== "hybrid") { status("transient needs a hybrid result", "err"); return; }
  let d;
  try {
    d = await api("GET", `/projects/${state.project}/transient`
      + `?period=${numOf("tr-period")}&pulse=${numOf("tr-pulse")}&compliance=${numOf("tr-compliance")}`);
  } catch (e) {
    status(/409/.test(e.message) ? "run a simulation first" : e.message, "err");
    return;
  }
  drawTransient($("tr-chart"), d);
  const damp = d.inlet_pulsatility > 0 ? (1 - d.tissue_pulsatility / d.inlet_pulsatility) * 100 : 0;
  $("tr-info").textContent =
    `R=${ttdFmt(d.R)} · C=${ttdFmt(d.C)} · tissue-flow pulsatility ${(d.tissue_pulsatility * 100).toFixed(0)}%`
    + ` vs inlet ${(d.inlet_pulsatility * 100).toFixed(0)}% (${damp.toFixed(0)}% damped)`;
  status("pulsatile response ready", "ok");
}

function drawTransient(el, d) {
  const ctx = el.getContext("2d"), W = el.width, H = el.height;
  ctx.clearRect(0, 0, W, H);
  const t = d.t || [];
  if (t.length < 2) return;
  const pad = { l: 6, r: 6, t: 8, b: 14 };
  const rel = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length || 1; return a.map((x) => (x - m) / m); };
  const pin = rel(d.p_in), q = rel(d.q_tissue);
  let amp = 1e-9;
  for (const s of [pin, q]) for (const v of s) amp = Math.max(amp, Math.abs(v));
  const X = (i) => pad.l + (i / (t.length - 1)) * (W - pad.l - pad.r);
  const Y = (v) => H / 2 - (v / amp) * ((H - pad.t - pad.b) / 2);
  ctx.strokeStyle = getCss("--line");                        // mean line
  ctx.beginPath(); ctx.moveTo(pad.l, H / 2); ctx.lineTo(W - pad.r, H / 2); ctx.stroke();
  const line = (s, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    s.forEach((v, i) => { const x = X(i), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  line(pin, getCss("--art"));                                // inlet pressure (relative to mean)
  line(q, getCss("--accent"));                               // tissue flow (relative to mean)
  ctx.fillStyle = getCss("--muted"); ctx.font = "9px -apple-system, sans-serif";
  ctx.fillText("— inlet p   — tissue flow (rel. to mean)", pad.l, H - 4);
}

// voxel-wise parametric map (Fp / vp) rendered on the main result canvas
export async function loadKineticMap(which) {
  requireProject();
  if (state.resultModel !== "hybrid") { status("maps need the hybrid tracer", "err"); return; }
  let f;
  try { f = await api("GET", `/projects/${state.project}/kinetics/maps?field=${which}`); }
  catch (e) { status(/409/.test(e.message) ? "compute the ▶ Tracer first" : e.message, "err"); return; }
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  stopFlow();
  state.fieldData = { ...f, vmin, vmax };
  state.field = which === "Fp" ? "Fp (perfusion)" : "BV (blood volume)";
  redraw();
  status(`${which} map`, "ok");
}

function drawTKM(el, d) {
  const ctx = el.getContext("2d"), W = el.width, H = el.height;
  ctx.clearRect(0, 0, W, H);
  const t = d.times || [];
  if (t.length < 2) return;
  const pad = { l: 6, r: 6, t: 8, b: 14 };
  const series = [d.aif, d.measured, d.fitted];
  let vmax = 0;
  for (const s of series) if (s) for (const v of s) if (v != null && v > vmax) vmax = v;
  vmax = vmax || 1;
  const X = (i) => pad.l + (i / (t.length - 1)) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - (v / vmax) * (H - pad.t - pad.b);
  const line = (s, color, dash = []) => {
    if (!s) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dash); ctx.beginPath();
    let started = false;
    s.forEach((v, i) => { if (v == null) return; const x = X(i), y = Y(v); started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true; });
    ctx.stroke(); ctx.setLineDash([]);
  };
  line(d.aif, getCss("--muted"), [3, 2]);        // arterial input (faint dashed)
  line(d.measured, getCss("--accent"));           // tissue curve
  line(d.fitted, getCss("--art"));                // fitted TKM
  ctx.strokeStyle = getCss("--line");
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  ctx.fillStyle = getCss("--muted"); ctx.font = "9px -apple-system, sans-serif";
  ctx.fillText("— tissue  — fit  ·· AIF", pad.l, H - 4);
}

function drawTTD(el, d) {
  const ctx = el.getContext("2d"), W = el.width, H = el.height;
  const { centres, density } = d.histogram;
  if (!centres.length) return;
  const pad = { l: 8, r: 8, t: 12, b: 18 };
  const x0 = centres[0], x1 = centres[centres.length - 1] || 1;
  const dmax = Math.max(...density) || 1;
  const bw = (W - pad.l - pad.r) / centres.length;
  const plotH = H - pad.t - pad.b;
  ctx.fillStyle = getCss("--accent");
  centres.forEach((_, i) => {
    const h = (density[i] / dmax) * plotH;
    ctx.fillRect(pad.l + i * bw, H - pad.b - h, Math.max(1, bw - 1), h);
  });
  // MTT marker (dashed red line + label)
  const mttX = pad.l + ((d.mtt - x0) / ((x1 - x0) || 1)) * (W - pad.l - pad.r);
  ctx.strokeStyle = getCss("--art"); ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(mttX, pad.t); ctx.lineTo(mttX, H - pad.b); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = getCss("--art"); ctx.font = "10px -apple-system, sans-serif";
  ctx.fillText("MTT", Math.min(mttX + 3, W - 28), pad.t + 8);
  // baseline + axis label
  ctx.strokeStyle = getCss("--line");
  ctx.beginPath(); ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
  ctx.fillStyle = getCss("--muted");
  ctx.fillText("τ (transit time) →", pad.l, H - 5);
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
  state.runNotes = s.notes || [];
  state.resultModel = s.model || "hybrid";
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
  const stats = [
    ["arterial inflow", s.arterial_inflow], ["venous outflow", s.venous_outflow],
    ["pressure max", s.pressure_max], ["flux mean", s.flux_mean],
  ];
  if (s.model === "direct") stats.push(["connections", s.n_connections]);
  if (s.mtt != null) stats.push(["MTT", s.mtt], ["CTH", s.cth], ["RD", s.rd]);   // transit-time metrics
  $("summary").innerHTML = stats
    .map(([k, v]) => `<div class="stat"><b>${fmt(v)}</b><span>${k}</span></div>`).join("");
}

export async function loadNetwork() {
  try { state.network = await api("GET", `/projects/${state.project}/network`); }
  catch { state.network = null; }
}

export async function loadField(name) {
  state.field = name;
  const f = await api("GET", `/projects/${state.project}/results/field/${name}`);
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  state.fieldData = { ...f, vmin, vmax };
  redraw();
}

// ── tracer flow video ─────────────────────────────────────────────────────────
export function stopTimer() {
  if (state.flow.timer) { clearInterval(state.flow.timer); state.flow.timer = null; }
}

export async function computeFlow() {
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
  fitKinetics().catch(() => {});    // also fit a TKM to the bolus (best-effort)
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
export function pauseFlow() { stopTimer(); $("flow-play").textContent = "▶"; }
export function toggleFlowPlay() { state.flow.timer ? pauseFlow() : playFlow(); }
export function stopFlow() {
  pauseFlow(); state.flow.active = false;
  ["flow-play", "flow-scrub", "flow-time"].forEach((id) => ($(id).hidden = true));
}
export function syncFlowUI() {
  $("flow-scrub").value = state.flow.idx;
  $("flow-time").textContent = `t = ${state.tracer.times[state.flow.idx].toFixed(2)}`;
}
