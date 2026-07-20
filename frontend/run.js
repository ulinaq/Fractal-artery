// perfusion 2-D workspace — grow / solve / results / tracer.
// Kicks off grow & run jobs, loads the result view (Darcy field or the
// direct-connection network), and drives the tracer flow animation.
//
// Cross-module refs (app wires the buttons; edit reuses loadNetwork/stopFlow) are
// resolved by ES-module live bindings — every call happens at runtime.
import { $, api, numOf, state, status } from "./core.js";
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
      rho_c: numOf("rho-c") });
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
}

async function loadDirect() {
  try { state.direct = await api("GET", `/projects/${state.project}/direct`); }
  catch { state.direct = null; }
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
