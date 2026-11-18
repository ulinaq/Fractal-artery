// perfusion — fit the model to a measured DCE-MRI curve (several inputs at once),
// shared by the 2-D workspace and the 3-D viewer. Builds its own panel into a
// container: paste CSV (t, tissue, aif) or take this project's own tracer curve,
// tick the inputs to fit (+ bounds), Fit → a job of full forward simulations, then
// the fitted values ± standard errors, R², warnings and measured-vs-model curves.

const INPUTS = [
  ["p_in", "p_in (flow)", 1, 50, true],
  ["porosity", "porosity (blood volume)", 0.02, 1, true],
  ["permeability", "K (bed)", 0.1, 10, false],
  ["mu", "μ (viscosity)", 0.1, 10, false],
];

const fmt = (v) => (v == null || !Number.isFinite(v) ? "–"
  : Math.abs(v) < 1e-3 || Math.abs(v) > 1e4 ? v.toExponential(2) : v.toPrecision(4));

export function mountDceFit(box, { projectId, prefix }) {
  const id = (s) => `${prefix}${s}`;
  box.innerHTML = `
    <p class="muted">Fit several model inputs at once so the simulated tissue curve
      matches a measured DCE-MRI curve. Paste CSV rows <code>t, tissue, aif</code>, or use
      this project's own tracer curve (🌊 flow first) to try it.</p>
    <textarea id="${id("csv")}" rows="4" placeholder="t, tissue, aif&#10;0, 0, 0&#10;1, 0.02, 0.4&#10;…"></textarea>
    <div class="row">
      <button id="${id("own")}" class="ghost">use this project's tracer curve</button>
    </div>
    <div class="dce-params">${INPUTS.map(([k, label, lo, hi, on]) => `
      <label class="check"><input type="checkbox" id="${id(`on-${k}`)}" ${on ? "checked" : ""} /> ${label}</label>
      <span><input class="mini" id="${id(`lo-${k}`)}" type="number" step="any" value="${lo}" /> –
            <input class="mini" id="${id(`hi-${k}`)}" type="number" step="any" value="${hi}" /></span>`).join("")}
    </div>
    <button id="${id("go")}">Fit to the curve</button>
    <p class="muted" id="${id("info")}"></p>
    <div id="${id("out")}"></div>
    <canvas id="${id("chart")}" class="dce-chart" width="288" height="150" hidden></canvas>`;
  const $ = (s) => document.getElementById(id(s));

  function parseCsv(text) {
    const rows = text.split(/\n/).map((l) => l.trim()).filter((l) => l && !/^[a-z#]/i.test(l))
      .map((l) => l.split(/[,;\t ]+/).map(Number));
    if (rows.length < 4 || rows.some((r) => r.length < 3 || r.slice(0, 3).some((v) => !Number.isFinite(v)))) {
      throw new Error("need at least 4 rows of three numbers: t, tissue, aif");
    }
    return { times: rows.map((r) => r[0]), tissue: rows.map((r) => r[1]), aif: rows.map((r) => r[2]) };
  }

  async function useOwn() {
    const r = await fetch(`/projects/${projectId()}/kinetics`);
    if (!r.ok) { $("info").textContent = "compute the 🌊 flow (tracer) first"; return; }
    const k = await r.json();
    $("csv").value = "t, tissue, aif\n" + k.times.map((t, i) =>
      `${+t.toPrecision(6)}, ${+(k.measured[i] ?? 0).toPrecision(6)}, ${+(k.aif[i] ?? 0).toPrecision(6)}`).join("\n");
    if (Number($("hi-porosity").value) <= 1) $("hi-porosity").value = 2;   // its bed has porosity 1
    $("info").textContent = `${k.times.length} samples from the simulated tracer — the fit should `
      + "recover this project's inputs (p_in, porosity 1; its porosity bound is widened to 2)";
  }

  function draw(fit) {
    const cv = $("chart"), ctx = cv.getContext("2d");
    cv.hidden = false;
    const { times, tissue, aif } = fit.input;
    const W = cv.width, H = cv.height, pad = 22;
    const tmax = times[times.length - 1] || 1;
    const ymax = Math.max(...tissue, ...fit.model, 1e-12);
    const amax = Math.max(...aif, 1e-12);
    const X = (t) => pad + (t / tmax) * (W - 2 * pad), Y = (v, m) => H - pad - (v / m) * (H - 2 * pad);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#cbd5e1"; ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();
    const line = (vals, m, color, dash) => {
      ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
      vals.forEach((v, i) => (i ? ctx.lineTo(X(times[i]), Y(v, m)) : ctx.moveTo(X(times[i]), Y(v, m))));
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(aif, amax, "#94a3b8", [4, 3]);                       // AIF (own scale)
    line(fit.model, ymax, "#2563eb");                          // fitted model
    ctx.fillStyle = "#c2410c";                                 // measured samples
    tissue.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(times[i]), Y(v, ymax), 2.2, 0, 7); ctx.fill(); });
    ctx.fillStyle = "#64748b"; ctx.font = "10px system-ui";
    ctx.fillText("● measured  — model  ┄ AIF (scaled)", pad, 12);
  }

  function show(fit) {
    const rows = fit.names.map((n) => `<tr><td>${n}</td><td>${fmt(fit.params[n])}</td>`
      + `<td>± ${fmt(fit.std[n])}</td></tr>`).join("");
    const warn = (fit.warnings || []).map((w) => `<p class="warn">⚠ ${w}</p>`).join("");
    $("out").innerHTML = `<table class="cmp"><tr><th>input</th><th>fitted</th><th>± 1σ</th></tr>${rows}</table>`
      + `<p class="muted">R² = ${fit.r2.toFixed(4)} · ${fit.n_eval} simulations (${fit.n_solve} flow solves)</p>${warn}`;
    draw(fit);
  }

  async function go() {
    let data;
    try { data = parseCsv($("csv").value); } catch (e) { $("info").textContent = e.message; return; }
    const params = {};
    for (const [k] of INPUTS) {
      if ($(`on-${k}`).checked) params[k] = [Number($(`lo-${k}`).value), Number($(`hi-${k}`).value)];
    }
    if (!Object.keys(params).length) { $("info").textContent = "tick at least one input to fit"; return; }
    $("go").disabled = true;
    $("info").textContent = "fitting… (each step is a full simulation)";
    try {
      const r = await fetch(`/projects/${projectId()}/fit-dce`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...data, params }) });
      const job = await r.json();
      if (!r.ok) throw new Error(typeof job.detail === "string" ? job.detail : "fit failed");
      let st = job;
      while (st.state !== "done" && st.state !== "error") {
        await new Promise((res) => setTimeout(res, 400));
        st = await (await fetch(`/jobs/${job.id}`)).json();
      }
      if (st.state === "error") throw new Error(st.error || "fit failed");
      show(await (await fetch(`/projects/${projectId()}/fit-dce`)).json());
      $("info").textContent = "";
    } catch (e) { $("info").textContent = e.message; } finally { $("go").disabled = false; }
  }

  async function load() {                                // the last fit, if any
    const r = await fetch(`/projects/${projectId()}/fit-dce`);
    if (r.ok) {
      const fit = await r.json();
      const { times, tissue, aif } = fit.input;
      $("csv").value = "t, tissue, aif\n" + times.map((t, i) => `${t}, ${tissue[i]}, ${aif[i]}`).join("\n");
      show(fit);
    } else { $("out").innerHTML = ""; $("chart").hidden = true; }
  }

  $("own").addEventListener("click", () => useOwn().catch((e) => { $("info").textContent = e.message; }));
  $("go").addEventListener("click", go);
  return { load };
}
