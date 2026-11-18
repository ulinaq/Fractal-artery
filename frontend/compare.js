// perfusion — run comparison (A/B), shared by the 2-D workspace and the 3-D viewer.
// Keep the current result as a named baseline (POST /snapshots), change something
// (a lesion, the vessels, p_in…), re-run, then compare: a table of what changed
// (GET /compare) and signed difference maps (Δ = current − baseline).

const LABELS = {
  arterial_inflow: "arterial inflow", venous_outflow: "venous outflow",
  pressure_max: "pressure max", flux_mean: "flux mean", mtt: "MTT", cth: "CTH", rd: "RD",
  tof_max: "TOF max", flux_range: "flux range", pressure_min: "pressure min",
  flux_std: "flux std", tof_mean: "TOF mean", tof_std: "TOF std",
  viscosity_factor_min: "viscosity × min", viscosity_factor_max: "viscosity × max",
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
  return `<p class="muted">vs <b>${c.snapshot.name}</b></p>`
    + `<table class="cmp"><tr><th></th><th>baseline</th><th>now</th><th>Δ</th></tr>${rows}${fields}</table>${note}`;
}

// a batch experiment as a sweep table: one row per value (click a row to pick it as
// the baseline to compare against)
export function sweepHtml(e) {
  const cols = [["arterial_inflow", "inflow"], ["mtt", "MTT"], ["cth", "CTH"], ["flux_mean", "flux mean"]];
  const head = cols.map(([, l]) => `<th>${l}</th>`).join("");
  const rows = e.rows.map((r) => `<tr data-snap="${r.snapshot}" title="compare against this run">`
    + `<td>${fmt(r.value)}</td>${cols.map(([k]) => `<td>${fmt(r.summary && r.summary[k])}</td>`).join("")}</tr>`)
    .join("");
  return `<p class="muted"><b>${e.name}</b> <button class="ghost mini-btn" data-del-exp="${e.id}" `
    + `title="delete this experiment and its runs">✕</button></p>`
    + `<table class="cmp sweep"><tr><th>${e.param}</th>${head}</tr>${rows}</table>`;
}

// Wire a compare panel. `ids` names the elements: name (text input), keep (button),
// select, run (button), del (button), out (div), maps (row of Δ-map buttons with
// data-delta="pressure|flux|tof"), and for batch experiments bparam (select), bvals
// (text: the values), brun (button), bout (div). `onMap(name, sid)` shows a Δ map.
export function createCompare({ projectId, ids, onMap, onInfo }) {
  const $ = (id) => document.getElementById(id);
  const el = Object.fromEntries(Object.entries(ids).map(([k, id]) => [k, $(id)]));
  const info = (msg, err) => onInfo && onInfo(msg, err);

  async function call(method, path, body) {
    const r = await fetch(`/projects/${projectId()}/${path}`, {
      method, headers: body ? { "content-type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined });
    const j = r.status === 204 ? {} : await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : `${path} failed (${r.status})`);
    return j;
  }

  async function refresh(selectId) {
    if (!projectId()) return;
    const { snapshots } = await call("GET", "snapshots");
    const keep = selectId || el.select.value;
    el.select.innerHTML = snapshots.length
      ? snapshots.map((s) => `<option value="${s.id}">${s.name}</option>`).join("")
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

  async function refreshBatch() {
    if (!el.bout || !projectId()) return;
    const { experiments } = await call("GET", "experiments");
    el.bout.innerHTML = experiments.map(sweepHtml).join("");
    el.bout.querySelectorAll("tr[data-snap]").forEach((tr) => tr.addEventListener("click", async () => {
      await refresh(tr.dataset.snap);
      await compare();
    }));
    el.bout.querySelectorAll("[data-del-exp]").forEach((b) => b.addEventListener("click", async () => {
      try {
        await call("DELETE", `experiments/${b.dataset.delExp}`);
        await refresh(); await refreshBatch();
      } catch (e) { info(e.message, true); }
    }));
  }

  async function runBatch() {
    const values = el.bvals.value.split(/[,\s]+/).filter(Boolean).map(Number);
    if (values.length < 2 || values.some((v) => !Number.isFinite(v))) {
      info("list at least two numbers, e.g. 5, 10, 20", true); return;
    }
    el.brun.disabled = true;
    info(`running ${values.length} solves…`);
    try {
      const job = await call("POST", "experiments", { param: el.bparam.value, values });
      let st = job;
      while (st.state !== "done" && st.state !== "error") {
        await new Promise((r) => setTimeout(r, 400));
        st = await (await fetch(`/jobs/${job.id}`)).json();
      }
      if (st.state === "error") { info(st.error || "experiment failed", true); return; }
      await refresh(); await refreshBatch();
      info(`${values.length} runs kept — click a row to compare against it`);
    } catch (e) { info(e.message, true); } finally { el.brun.disabled = false; }
  }

  if (el.brun) el.brun.addEventListener("click", runBatch);
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
