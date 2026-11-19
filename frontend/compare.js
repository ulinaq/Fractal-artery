// perfusion — run comparison (A/B), shared by the 2-D workspace and the 3-D viewer.
// Keep the current result as a named baseline (POST /snapshots), change something
// (a lesion, the vessels, p_in…), re-run, then compare: a table of what changed
// (GET /compare) and signed difference maps (Δ = current − baseline).

import { escapeHtml, fetchJSON, pollJob } from "./net.js";
import { viridisCss } from "./palette.js";

const LABELS = {
  arterial_inflow: "arterial inflow", venous_outflow: "venous outflow",
  pressure_max: "pressure max", flux_mean: "flux mean", mtt: "MTT", cth: "CTH", rd: "RD",
  tof_max: "TOF max", flux_range: "flux range", pressure_min: "pressure min",
  flux_std: "flux std", tof_mean: "TOF mean", tof_std: "TOF std",
  viscosity_factor_min: "viscosity × min", viscosity_factor_max: "viscosity × max",
  plugged_vessels: "plugged vessels",
};
// network sizes / solver counts: listed only when they changed (e.g. after regrowing a tree)
const STRUCTURE = /(_nodes|_segments|_iterations)$/;

const fmt = (v) => (v == null || !Number.isFinite(v) ? "–"
  : Math.abs(v) < 1e-3 || Math.abs(v) > 1e4 ? v.toExponential(2) : v.toPrecision(4));

// the comparison as HTML: per-quantity change table + per-field change rows
export function compareHtml(c) {
  const pct = (r) => (r == null ? "" : `${r > 0 ? "+" : ""}${(100 * r).toFixed(1)}%`);
  const cls = (r) => (r == null || Math.abs(r) < 1e-9 ? "" : r < 0 ? "dn" : "up");
  const rows = (c.summary || []).filter((r) => !(STRUCTURE.test(r.key) && r.diff === 0))
    .map((r) => `<tr><td>${LABELS[r.key] || r.key.replaceAll("_", " ")}</td>`
    + `<td>${fmt(r.baseline)}</td><td>${fmt(r.current)}</td>`
    + `<td class="${cls(r.rel ?? r.diff)}">${pct(r.rel) || fmt(r.diff)}</td></tr>`).join("");
  const fields = Object.entries(c.fields || {}).filter(([, f]) => f.cells)
    .map(([k, f]) => `<tr><td>${k} (mean)</td><td>${fmt(f.mean_baseline)}</td>`
      + `<td>${fmt(f.mean_current)}</td><td class="${cls(f.rel_change)}">${pct(f.rel_change)}</td></tr>`)
    .join("");
  const note = c.same_grid ? "" : '<p class="muted">The grid changed, so only the summary compares.</p>';
  return `<p class="muted">vs <b>${escapeHtml(c.snapshot.name)}</b></p>`
    + `<table class="cmp"><tr><th></th><th>baseline</th><th>now</th><th>Δ</th></tr>${rows}${fields}</table>${note}`;
}

// a batch experiment as a sweep table: one row per value (click a row to pick it as
// the baseline to compare against)
const SWEEP_COLS = [["arterial_inflow", "inflow"], ["mtt", "MTT"], ["cth", "CTH"], ["flux_mean", "flux mean"]];

// elasticity line: "inflow ∝ p_in^1.00 · K^0.02" (d ln y / d ln input, from the sweep)
function sensitivityHtml(e, key, label) {
  const s = e.sensitivity && e.sensitivity[key];
  if (!s) return "";
  const parts = Object.entries(s).filter(([, v]) => v != null)
    .map(([p, v]) => `${p}<sup>${v.toFixed(2)}</sup>`);
  return parts.length
    ? `<p class="muted sens" title="elasticity d ln y / d ln input — 1: proportional, 0: insensitive">`
      + `${label} ∝ ${parts.join(" · ")}</p>` : "";
}

// A one-input sweep → a table (one row per value); a two-input grid → a matrix of
// `metric` (rows: param, columns: param2) shaded on viridis. Every run is clickable
// (data-snap) to compare against it.
export function sweepHtml(e, metric = "arterial_inflow") {
  const head = `<p class="muted"><b>${escapeHtml(e.name)}</b> <button class="ghost mini-btn" data-del-exp="${escapeHtml(e.id)}" `
    + `title="delete this experiment and its runs">✕</button></p>`;
  if (!e.param2) {
    const cols = SWEEP_COLS.map(([, l]) => `<th>${l}</th>`).join("");
    const rows = e.rows.map((r) => `<tr data-snap="${r.snapshot}" title="compare against this run">`
      + `<td>${fmt(r.value)}</td>${SWEEP_COLS.map(([k]) => `<td>${fmt(r.summary && r.summary[k])}</td>`).join("")}</tr>`)
      .join("");
    return head + `<table class="cmp sweep"><tr><th>${e.param}</th>${cols}</tr>${rows}</table>`
      + sensitivityHtml(e, "arterial_inflow", "inflow");
  }
  const v1 = [...new Set(e.rows.map((r) => r.value))].sort((a, b) => a - b);
  const v2 = [...new Set(e.rows.map((r) => r.value2))].sort((a, b) => a - b);
  const cell = new Map(e.rows.map((r) => [`${r.value}|${r.value2}`, r]));
  const val = (r) => (r && r.summary ? r.summary[metric] : null);
  const all = e.rows.map(val).filter((v) => v != null && Number.isFinite(v));
  const lo = Math.min(...all), hi = Math.max(...all);
  const label = (SWEEP_COLS.find(([k]) => k === metric) || [metric, metric])[1];
  const pick = `<select class="mini" data-metric-exp="${e.id}">${SWEEP_COLS.map(([k, l]) =>
    `<option value="${k}" ${k === metric ? "selected" : ""}>${l}</option>`).join("")}</select>`;
  const ax = (v) => String(+v.toPrecision(4));             // axis values: 0.5, not 0.5000
  const rows = v1.map((a) => `<tr><th>${ax(a)}</th>${v2.map((b) => {
    const r = cell.get(`${a}|${b}`), v = val(r);
    if (!r) return "<td></td>";
    const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    const bg = v == null ? "" : `background:${viridisCss(t)};color:${t > 0.55 ? "#111" : "#fff"}`;
    return `<td data-snap="${r.snapshot}" style="${bg}" title="${e.param} = ${a}, ${e.param2} = ${b} — compare against this run">${fmt(v)}</td>`;
  }).join("")}</tr>`).join("");
  return head + `<p class="muted grid-head">${pick} · rows ${e.param}, columns ${e.param2}</p>`
    + `<table class="cmp sweep grid"><tr><th>${e.param} ↓ ${e.param2} →</th>${v2.map((b) => `<th>${ax(b)}</th>`).join("")}</tr>${rows}</table>`
    + sensitivityHtml(e, metric, label);
}

// Wire a compare panel. `ids` names the elements: name (text input), keep (button),
// select, run (button), del (button), out (div), maps (row of Δ-map buttons with
// data-delta="pressure|flux|tof"), and for batch experiments bparam (select), bvals
// (text: the values), optional bparam2/bvals2 (a second input → a grid), brun
// (button), bout (div). `onMap(name, sid)` shows a Δ map.
export function createCompare({ projectId, ids, onMap, onInfo }) {
  const $ = (id) => document.getElementById(id);
  const el = Object.fromEntries(Object.entries(ids).map(([k, id]) => [k, $(id)]));
  const info = (msg, err) => onInfo && onInfo(msg, err);

  async function call(method, path, body) {
    return (await fetchJSON(method, `/projects/${projectId()}/${path}`, body)) || {};
  }

  async function refresh(selectId) {
    if (!projectId()) return;
    const { snapshots } = await call("GET", "snapshots");
    const keep = selectId || el.select.value;
    el.select.innerHTML = snapshots.length
      ? snapshots.map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.name)}</option>`).join("")
      : '<option value="">no baseline yet</option>';
    if (snapshots.some((s) => s.id === keep)) el.select.value = keep;
    el.select.disabled = el.run.disabled = el.del.disabled = !snapshots.length;
  }

  async function keep() {
    try {
      const s = await call("POST", "snapshots", { name: el.name.value || null });
      el.name.value = "";
      await refresh(s.id);
      info(`kept "${s.name}" — change something, re-run, then compare`);
    } catch (e) { info(e.message, true); }
  }

  async function compare() {
    if (!el.select.value) return;
    try {
      const c = await call("GET", `compare?snapshot=${el.select.value}`);
      el.out.innerHTML = compareHtml(c);
      el.maps.hidden = !c.same_grid;
      info("");
    } catch (e) { el.out.innerHTML = ""; el.maps.hidden = true; info(e.message, true); }
  }

  async function remove() {
    if (!el.select.value) return;
    try {
      await call("DELETE", `snapshots/${el.select.value}`);
      el.out.innerHTML = ""; el.maps.hidden = true;
      await refresh(); await refreshBatch();
    } catch (e) { info(e.message, true); }
  }

  const metricOf = {};                       // grid experiment id -> shown quantity
  let experiments = [];

  function renderBatch() {
    el.bout.innerHTML = experiments.map((e) => sweepHtml(e, metricOf[e.id])).join("");
    el.bout.querySelectorAll("[data-snap]").forEach((c) => c.addEventListener("click", async () => {
      await refresh(c.dataset.snap);
      await compare();
    }));
    el.bout.querySelectorAll("[data-metric-exp]").forEach((sel) => sel.addEventListener("change", () => {
      metricOf[sel.dataset.metricExp] = sel.value;
      renderBatch();
    }));
    el.bout.querySelectorAll("[data-del-exp]").forEach((b) => b.addEventListener("click", async () => {
      try {
        await call("DELETE", `experiments/${b.dataset.delExp}`);
        await refresh(); await refreshBatch();
      } catch (e) { info(e.message, true); }
    }));
  }

  async function refreshBatch() {
    if (!el.bout || !projectId()) return;
    ({ experiments } = await call("GET", "experiments"));
    renderBatch();
  }

  const parse = (input) => input.value.split(/[,\s]+/).filter(Boolean).map(Number);
  const badList = (v) => v.length < 2 || v.some((x) => !Number.isFinite(x));

  async function runBatch() {
    const values = parse(el.bvals);
    if (badList(values)) { info("list at least two numbers, e.g. 5, 10, 20", true); return; }
    const body = { param: el.bparam.value, values };
    if (el.bparam2 && el.bparam2.value) {          // a second input → a grid
      const values2 = parse(el.bvals2);
      if (badList(values2)) { info("list at least two numbers for the second input", true); return; }
      if (el.bparam2.value === body.param) { info("pick two different inputs for a grid", true); return; }
      Object.assign(body, { param2: el.bparam2.value, values2 });
    }
    const n = values.length * (body.values2 ? body.values2.length : 1);
    if (n > 48) { info(`a grid is at most 48 solves (${n} asked)`, true); return; }
    el.brun.disabled = true;
    info(`running ${n} solves…`);
    try {
      const job = await call("POST", "experiments", body);
      const st = await pollJob(job.id);
      if (st.state === "error") { info(st.error || "experiment failed", true); return; }
      await refresh(); await refreshBatch();
      info(`${n} runs kept — click a ${body.param2 ? "cell" : "row"} to compare against it`);
    } catch (e) { info(e.message, true); } finally { el.brun.disabled = false; }
  }

  if (el.brun) el.brun.addEventListener("click", runBatch);
  if (el.bparam2) {                            // the second input's values only with one chosen
    const sync = () => { el.bvals2.hidden = !el.bparam2.value; };
    el.bparam2.addEventListener("change", sync); sync();
  }
  el.keep.addEventListener("click", keep);
  el.run.addEventListener("click", compare);
  el.del.addEventListener("click", remove);
  el.select.addEventListener("change", () => { el.out.innerHTML = ""; el.maps.hidden = true; });
  el.maps.querySelectorAll("[data-delta]").forEach((b) => b.addEventListener("click",
    () => onMap(b.dataset.delta, el.select.value)));
  return {
    refresh: async (id) => { await refresh(id); await refreshBatch(); },
    compare, selected: () => el.select.value,
  };
}
