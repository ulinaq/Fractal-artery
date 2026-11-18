// Uncertainty-quantification result tables (POST /projects/{id}/uq), shared by the
// 2-D workspace and the 3-D viewer: Pearson sensitivities or Sobol S₁ / Sₜ bars.

/** HTML for a UQ response — ``method: "sobol"`` or ``"correlation"``. */
export function uqHtml(d) {
  return d.method === "sobol" ? sobolHtml(d) : correlationHtml(d);
}

function sobolHtml(d) {
  const rows = d.qoi_names.map((q) => {
    const s1 = d.first_order[q], st = d.total_effect[q];
    const bars = d.names.map((p) => {
      const w1 = Math.round(s1[p] * 60), wt = Math.round(st[p] * 60);
      return `<tr><td>${p}</td>`
        + `<td><span class="bar" style="width:${wt}px;background:#94a3b8"></span>`
        + `<span class="bar" style="width:${w1}px;background:var(--accent);margin-left:-${wt}px"></span></td>`
        + `<td>S₁ ${s1[p].toFixed(2)} · Sₜ ${st[p].toFixed(2)}</td></tr>`;
    }).join("");
    return `<div><b>${q}</b> <span class="muted">(fit R²=${d.fit_r2[q].toFixed(2)})</span>`
      + `<table>${bars}</table></div>`;
  }).join("");
  return `<p class="muted">${d.n_samples} samples · Sobol variance share (S₁ inner, Sₜ outer bar)</p>${rows}`;
}

function correlationHtml(d) {
  const fmt = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(2) : v.toFixed(3);
  const rows = d.qoi_names.map((q) => {
    const sens = d.sensitivity[q];
    const bars = d.names.map((p) => {
      const r = sens[p], w = Math.round(Math.abs(r) * 60);
      const col = r < 0 ? "#ef4444" : "var(--accent)";
      return `<tr><td>${p}</td><td><span class="bar" style="width:${w}px;background:${col}"></span></td>`
        + `<td>${r >= 0 ? "+" : ""}${r.toFixed(2)}</td></tr>`;
    }).join("");
    return `<div><b>${q}</b>: ${fmt(d.mean[q])} ± ${fmt(d.std[q])} `
      + `<span class="muted">[${fmt(d.p05[q])}, ${fmt(d.p95[q])}]</span>`
      + `<table>${bars}</table></div>`;
  }).join("");
  return `<p class="muted">${d.n_samples} samples · sensitivity (Pearson r)</p>${rows}`;
}
