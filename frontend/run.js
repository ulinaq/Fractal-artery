// perfusion 2-D workspace — grow / solve / results / tracer.
// Kicks off grow & run jobs, loads the result view (Darcy field or the
// direct-connection network), and drives the tracer flow animation.
//
// Cross-module refs (app wires the buttons; edit reuses loadNetwork/stopFlow) are
// resolved by ES-module live bindings — every call happens at runtime.
import { $, api, getCss, numOf, state, status } from "./core.js";
import { redraw } from "./render.js";
import { uqHtml } from "./uqview.js";
import { loadEditable } from "./edit.js";
import { flushAll, recordHistory, requireProject, setMode } from "./app.js";

export async function grow() {
  requireProject();
  const target = document.querySelector("input[name=growtarget]:checked").value;
  await flushAll();                          // persist unsaved vessel/tissue edits
  status(`growing ${target}…`);
  const info = await api("POST", `/projects/${state.project}/grow`,
    { target, method: $("grow-method").value, n_points: numOf("grow-npts"),
      step: numOf("grow-step"), seed: numOf("seed"),
      close_loops: $("grow-method").value === "colonization" && $("grow-loops").checked });
  $("grow-info").textContent = `grown ${target}: ${info.n_segments} segments, ${info.n_terminals} terminals`
    + (info.n_loops ? `, ${info.n_loops} loops` : "");
  state.hasResult = false;         // grow invalidates any prior result server-side
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  recordHistory();
  status(`${target} tree grown`, "ok");
  redraw();
}

// Prune the outermost generations of the chosen tree (the inverse of Grow).
export async function prune() {
  requireProject();
  const target = document.querySelector("input[name=growtarget]:checked").value;
  await flushAll();                          // prune what's on screen, not the stale copy
  const steps = Math.max(1, Math.round(numOf("prune-steps")));
  const info = await api("POST", `/projects/${state.project}/prune`, { target, steps });
  $("prune-info").textContent = info.n_removed
    ? `pruned ${info.n_removed} ${target} segments → ${info.n_segments} left, ${info.n_terminals} ends`
    : `nothing to prune — only the ${target} trunk is left`;
  state.hasResult = false;
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  recordHistory();
  status(`${target} pruned by ${steps} step${steps > 1 ? "s" : ""}`, "ok");
  redraw();
}

// Anastomoses: join nearby branches of the chosen tree with collateral links (loops)
export async function addLoops() {
  requireProject();
  const target = document.querySelector("input[name=growtarget]:checked").value;
  await flushAll();
  const n_links = Math.max(1, Math.round(numOf("loop-n")));
  const info = await api("POST", `/projects/${state.project}/anastomose`, { target, n_links });
  $("loop-info").textContent = `added ${info.n_added} ${target} link${info.n_added > 1 ? "s" : ""}`
    + ` → ${info.n_loops} loop${info.n_loops === 1 ? "" : "s"}, ${info.n_terminals} ends kept`;
  state.hasResult = false;
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  recordHistory();
  status(`${target}: ${info.n_added} anastomoses added`, "ok");
  redraw();
}

// Monte-Carlo uncertainty quantification: run N re-solves over the checked input
// ranges, then show each output's mean±std and a sensitivity bar per input.
export async function runUQ() {
  requireProject();
  const params = {};
  if ($("uq-pin").checked) params.p_in = [numOf("uq-pin-lo"), numOf("uq-pin-hi")];
  if ($("uq-mu").checked) params.mu = [numOf("uq-mu-lo"), numOf("uq-mu-hi")];
  if ($("uq-rmin").checked) params.r_min_frac = [numOf("uq-rmin-lo"), numOf("uq-rmin-hi")];
  if (!Object.keys(params).length) { status("tick at least one uncertain input", "err"); return; }
  const sobol = $("uq-sobol").checked;
  const n = sobol ? Math.max(24, Math.round(numOf("uq-n"))) : Math.round(numOf("uq-n"));
  $("uq-out").textContent = `running ${n} solves…`;
  status(`uncertainty: ${n} solves…`);
  let d;
  try {
    d = await api("POST", `/projects/${state.project}/uq`,
      { n_samples: n, params, method: sobol ? "sobol" : "correlation" });
  } catch (e) { status(e.message, "err"); $("uq-out").textContent = e.message; return; }
  $("uq-out").innerHTML = uqHtml(d);
  status("uncertainty done", "ok");
}

// Horton–Strahler morphometry of the current tree, shown as ratios + a per-order table
export async function loadMorphometry(target) {
  requireProject();
  await flushAll();                          // analyse the edited tree, not the stale one
  $("morph-out").textContent = "analysing…";
  let d;
  try { d = await api("GET", `/projects/${state.project}/morphometry?target=${target}`); }
  catch (e) { status(e.message, "err"); $("morph-out").textContent = e.message; return; }
  if (!d.max_order) { $("morph-out").textContent = "no tree to analyse"; return; }
  const f = (v) => (v == null || Number.isNaN(v)) ? "–" : v.toFixed(2);
  const rows = Object.keys(d.counts).map((w) =>
    `<tr><td>ω=${w}</td><td>${d.counts[w]}</td><td>${f(d.mean_length[w])}</td>`
    + `<td>${f(d.mean_radius[w])}</td></tr>`).join("");
  $("morph-out").innerHTML =
    `<p class="muted">${target} · max order ${d.max_order} · `
    + `R<sub>B</sub>=${f(d.bifurcation_ratio)} · R<sub>L</sub>=${f(d.length_ratio)} · `
    + `R<sub>D</sub>=${f(d.diameter_ratio)}</p>`
    + `<table><tr><td>order</td><td>N</td><td>len</td><td>rad</td></tr>${rows}</table>`;
  status(`${target} morphometry`, "ok");
}

export async function adapt() {
  requireProject();
  await flushAll();                          // persist unsaved vessel/tissue edits
  status("adapting radii (wall-shear remodelling)…");
  let info;
  try {
    info = await api("POST", `/projects/${state.project}/adapt`,
      { target_exponent: numOf("adapt-target"), rate: numOf("adapt-rate") });
  } catch (e) { status(e.message, "err"); $("adapt-info").textContent = e.message; return; }
  $("adapt-info").textContent =
    `Murray exponent ${info.murray_start.toFixed(2)} → ${info.murray_final.toFixed(2)}`
    + ` in ${info.n_iter} iters${info.moved_toward_target ? "" : " (no improvement)"}`;
  state.hasResult = false;
  stopFlow(); state.tracer = null; state.fieldData = null;
  await Promise.all([loadNetwork(), loadEditable()]);
  recordHistory();
  status("radii adapted → Murray's law", "ok");
  redraw();
}

export async function run() {
  requireProject();
  await flushAll();                          // persist unsaved vessel/tissue edits
  const job = await api("POST", `/projects/${state.project}/run`,
    { model: $("run-model").value, nonlinear: $("nonlinear").checked,
      use_vessels: true, max_iter: numOf("max-iter"),
      rho_c: numOf("rho-c"), phase_separation: $("phase-sep").checked,
      elasticity: $("elasticity").checked, compliance: numOf("compliance"),
      shear_thinning: $("shear-thin").checked, carreau_lambda: numOf("carreau-lambda") });
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
  stopFlow(); state.tracer = null;           // drop any stale (field or network) tracer
  state.direct = null;
  if (state.resultModel === "direct") await loadDirect();
  else await loadField(state.field);
  await loadTransit();                        // h(τ) chart (hybrid) or a note (direct)
  if (state.onResultView) await state.onResultView();   // e.g. refresh the compare panel
}

// A signed difference map (current − a kept baseline), drawn with a diverging ramp
// symmetric about zero change.
export async function loadDeltaField(name, sid) {
  const f = await api("GET", `/projects/${state.project}/compare/field/${name}?snapshot=${sid}`);
  let m = 0;
  for (const row of f.data) for (const v of row) if (v !== null && Math.abs(v) > m) m = Math.abs(v);
  m = m || 1;
  stopFlow();
  document.querySelectorAll(".field").forEach((x) => x.classList.remove("active"));
  state.fieldData = { ...f, vmin: -m, vmax: m, delta: true };
  state.field = `Δ ${name}`;
  redraw();
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

// voxel-wise parametric map rendered on the main result canvas.
// method "semi" -> field Fp|vp; method "deconv" (SVD residue) -> field Fp|CBV|MTT.
const MAP_LABELS = {
  Fp: "Fp (perfusion)", vp: "BV (blood volume)",
  CBV: "CBV (blood volume)", MTT: "MTT (transit time)",
};
export async function loadKineticMap(which, method = "semi") {
  requireProject();
  if (state.resultModel !== "hybrid") { status("maps need the hybrid tracer", "err"); return; }
  const q = `field=${which}&method=${method}`;
  let f;
  try { f = await api("GET", `/projects/${state.project}/kinetics/maps?${q}`); }
  catch (e) { status(/409/.test(e.message) ? "compute the ▶ Tracer first" : e.message, "err"); return; }
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  stopFlow();
  state.fieldData = { ...f, vmin, vmax };
  state.field = MAP_LABELS[which] || which;
  redraw();
  status(`${which} map${method === "deconv" ? " (deconv)" : ""}`, "ok");
}

// multi-compartment Darcy field (capillary perfusion or a compartment pressure)
export async function loadMulticompartment(which) {
  requireProject();
  if (state.resultModel !== "hybrid") { status("multi-compartment needs a hybrid result", "err"); return; }
  const n = Math.round(numOf("mc-n")), beta = numOf("mc-beta");
  const field = which === "venule" ? `comp${n - 1}` : which === "arteriole" ? "comp0" : "perfusion";
  $("mc-info").textContent = "solving compartments…";
  let f;
  try { f = await api("GET", `/projects/${state.project}/multicompartment?field=${field}&n=${n}&beta=${beta}`); }
  catch (e) { status(e.message, "err"); $("mc-info").textContent = e.message; return; }
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  stopFlow();
  state.fieldData = { ...f, vmin, vmax };
  state.field = which === "perfusion" ? `capillary perfusion (N=${f.n})`
    : which === "arteriole" ? "arteriole pressure" : "venule pressure";
  redraw();
  $("mc-info").textContent = `N=${f.n}, β=${f.beta} — ${state.field}`;
  status(`multi-compartment ${which}`, "ok");
}

// steady oxygen/solute map (saturation or uptake) rendered on the main heatmap
export async function loadOxygen(which) {
  requireProject();
  if (state.resultModel !== "hybrid") { status("oxygen needs a hybrid result", "err"); return; }
  const k = numOf("o2-k"), diff = numOf("o2-diff");
  const q = `field=${which}&k=${k}&diffusivity=${diff}`;
  $("o2-info").textContent = "solving oxygen transport…";
  let f;
  try { f = await api("GET", `/projects/${state.project}/oxygen?${q}`); }
  catch (e) { status(e.message, "err"); $("o2-info").textContent = e.message; return; }
  let vmin = Infinity, vmax = -Infinity;
  for (const row of f.data) for (const v of row) if (v !== null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  stopFlow();
  state.fieldData = { ...f, vmin, vmax };
  state.field = which === "saturation" ? "O₂ saturation" : "O₂ uptake (k·C)";
  redraw();
  $("o2-info").textContent = `global OEF = ${(f.oef * 100).toFixed(1)}%  (k=${f.k})`;
  status(`O₂ ${which}`, "ok");
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
  if (s.viscosity_factor_max != null) {     // shear-thinning: Carreau viscosity range
    stats.push(["viscosity × min", s.viscosity_factor_min], ["viscosity × max", s.viscosity_factor_max]);
  }
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
  if (state.resultModel === "direct") { await computeNetworkFlow(); return; }
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
  ["flow-play", "flow-scrub", "flow-time", "flow-gif", "flow-mp4"].forEach((id) => ($(id).hidden = false));
  playFlow();
  fitKinetics().catch(() => {});    // also fit a TKM to the bolus (best-effort)
}

async function loadTracer() {
  const t = await api("GET", `/projects/${state.project}/tracer`);
  state.tracer = t;
  state.flow.idx = 0;
  $("flow-scrub").max = t.n_frames - 1;
  $("flow-gif").href = `/projects/${state.project}/tracer.gif`;
  $("flow-mp4").href = `/projects/${state.project}/tracer.mp4`;
}

// direct model: animate the bolus along the vascular network (transit-time based)
async function computeNetworkFlow() {
  const load = async () => {
    const t = await api("GET", `/projects/${state.project}/network-tracer`);
    state.tracer = { ...t, network: true };
    state.flow.idx = 0;
    $("flow-scrub").max = t.n_frames - 1;
  };
  if (!(state.tracer && state.tracer.network)) {
    try {
      await load();
    } catch {
      status("computing network transit… (solves + times the bolus)");
      const job = await api("POST", `/projects/${state.project}/tracer`, { n_frames: 48 });
      const final = await pollJob(job.id);
      if (final.state === "error") { status(final.error || "flow failed", "err"); return; }
      await load();
      status("network transit ready", "ok");
    }
  }
  state.flow.active = true;
  ["flow-play", "flow-scrub", "flow-time"].forEach((id) => ($(id).hidden = false));
  $("flow-gif").hidden = $("flow-mp4").hidden = true;   // no GIF/MP4 for the network animation
  playFlow();
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
