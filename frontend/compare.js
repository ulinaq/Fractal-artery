// perfusion — run comparison (A/B), shared by the 2-D workspace and the 3-D viewer.
// Keep the current result as a named baseline (POST /snapshots), change something
// (a lesion, the vessels, p_in…), re-run, then compare: a table of what changed
// (GET /compare) and signed difference maps (Δ = current − baseline).

const LABELS = {
  arterial_inflow: "arterial inflow", venous_outflow: "venous outflow",
  pressure_max: "pressure max", flux_mean: "flux mean", mtt: "MTT", cth: "CTH", rd: "RD",
  tof_max: "TOF max", flux_range: "flux range", pressure_min: "pressure min",
  flux_std: "flux std", tof_mean: "TOF mean", tof_std: "TOF std",
};
// network sizes: listed only when they changed (e.g. after regrowing a tree)
const STRUCTURE = /(_nodes|_segments)$/;

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

// Wire a compare panel. `ids` names the elements: name (text input), keep (button),
// select, run (button), del (button), out (div), maps (row of Δ-map buttons with
// data-delta="pressure|flux|tof"). `onMap(name, sid)` shows a difference map.
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
      await refresh();
    } catch (e) { info(e.message, true); }
  }

  el.keep.addEventListener("click", keep);
  el.run.addEventListener("click", compare);
  el.del.addEventListener("click", remove);
  el.select.addEventListener("change", () => { el.out.innerHTML = ""; el.maps.hidden = true; });
  el.maps.querySelectorAll("[data-delta]").forEach((b) => b.addEventListener("click",
    () => onMap(b.dataset.delta, el.select.value)));
  return { refresh, compare, selected: () => el.select.value };
}
