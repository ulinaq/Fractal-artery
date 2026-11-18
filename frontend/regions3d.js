// perfusion — lesions & hotspots on a 3-D project (spheres), for the 3-D viewer.
// A lesion is a low-permeability sphere (heterogeneous K in the Darcy bed); a hotspot
// is a growth-demand sphere (🌱 Grow samples more points there). Each is edited in
// the card (x, y, z, radius, value) or dragged in the screen plane by its centre
// handle (rotate the view to move it in depth), and saved through
// PUT /projects/{id}/lesions | /hotspots.

const KINDS = {
  lesion: { list: "lesions", value: "permeability", label: "lesion", unit: "K",
            def: 0.1, min: 0.001, color: "#f59e0b" },
  hotspot: { list: "hotspots", value: "weight", label: "hotspot", unit: "weight",
             def: 4, min: 1, color: "#c084fc" },
};

export function createRegions3d({ projectId, project, unproject, scene, onSaved, onInfo }) {
  const $ = (id) => document.getElementById(id);
  const st = { lesion: [], hotspot: [], drag: null, sel: null };

  async function load() {
    for (const kind of Object.keys(KINDS)) {
      const k = KINDS[kind];
      const r = await fetch(`/projects/${projectId}/${k.list}`);
      const j = r.ok ? await r.json() : {};
      st[kind] = (j[k.list] || []).filter((x) => x.center && x.center.length === 3);
    }
    renderList();
  }

  async function save(kind) {
    const k = KINDS[kind];
    const body = { [k.list]: st[kind].map((x) => ({ center: x.center.map(Number), radius: Number(x.radius),
                                                   [k.value]: Number(x[k.value]) })) };
    onInfo("saving…");
    const r = await fetch(`/projects/${projectId}/${k.list}`, {
      method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      onInfo(typeof j.detail === "string" ? j.detail : `saving the ${k.list} failed`, true);
      await load();
      return false;
    }
    onInfo(kind === "lesion" ? "lesions saved — re-run to solve with them"
                             : "hotspots saved — 🌱 Grow now favours them");
    if (kind === "lesion") await onSaved();          // the old result is invalidated
    return true;
  }

  // the tissue's centre and size (from the scene) for a new sphere
  function defaults() {
    const d = scene();
    const pts = (d && d.tissue_points) || [];
    const dom = d ? d.domain : [0, 1, 0, 1, 0, 1];
    const c = pts.length
      ? [0, 1, 2].map((a) => pts.reduce((s, p) => s + p[a], 0) / pts.length)
      : [0, 2, 4].map((a) => (dom[a] + dom[a + 1]) / 2);
    const size = Math.min(dom[1] - dom[0], dom[3] - dom[2], dom[5] - dom[4]);
    return { center: c.map((v) => Number(v.toFixed(2))), radius: Number((0.15 * size).toFixed(2)) };
  }

  async function add(kind) {
    const k = KINDS[kind];
    st[kind].push({ ...defaults(), [k.value]: k.def });
    st.sel = { kind, i: st[kind].length - 1 };
    renderList();
    await save(kind);
  }

  async function remove(kind, i) {
    st[kind].splice(i, 1);
    st.sel = null;
    renderList();
    await save(kind);
  }

  function renderList() {
    const box = $("v3-reg-list");
    if (!box) return;
    const rows = [];
    for (const kind of Object.keys(KINDS)) {
      const k = KINDS[kind];
      st[kind].forEach((x, i) => {
        const sel = st.sel && st.sel.kind === kind && st.sel.i === i;
        const num = (f, v, step) => `<input class="mini" data-kind="${kind}" data-i="${i}" data-f="${f}" `
          + `type="number" step="${step}" value="${v}" />`;
        rows.push(`<div class="reg-row${sel ? " sel" : ""}" data-kind="${kind}" data-i="${i}">`
          + `<span class="reg-tag" style="color:${k.color}">● ${k.label} ${i + 1}</span>`
          + `<label>x ${num("x", x.center[0], "any")}</label><label>y ${num("y", x.center[1], "any")}</label>`
          + `<label>z ${num("z", x.center[2], "any")}</label><label>r ${num("radius", x.radius, "any")}</label>`
          + `<label>${k.unit} ${num(k.value, x[k.value], "any")}</label>`
          + `<button class="ghost reg-del" data-kind="${kind}" data-i="${i}" title="delete">✕</button></div>`);
      });
    }
    box.innerHTML = rows.join("") || '<p class="muted">none yet</p>';
    box.querySelectorAll("input").forEach((el) => el.addEventListener("change", async () => {
      const { kind, i, f } = el.dataset, x = st[kind][Number(i)], v = Number(el.value);
      if (!Number.isFinite(v)) return;
      if (f === "x" || f === "y" || f === "z") x.center["xyz".indexOf(f)] = v;
      else if (f === "radius") x.radius = Math.max(1e-3, v);
      else x[f] = Math.max(KINDS[kind].min, v);
      st.sel = { kind, i: Number(i) };
      await save(kind);
      renderList();
    }));
    box.querySelectorAll(".reg-del").forEach((el) => el.addEventListener("click",
      () => remove(el.dataset.kind, Number(el.dataset.i))));
  }

  // screen radius of a sphere: project the centre and a point r away along screen-right
  function screenCircle(x) {
    const [sx, sy, depth] = project(...x.center);
    const a = unproject(sx, sy, depth), b = unproject(sx + 1, sy, depth);
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const n = Math.hypot(...u) || 1;
    const e = project(...x.center.map((c, j) => c + (x.radius * u[j]) / n));
    return { sx, sy, depth, r: Math.hypot(e[0] - sx, e[1] - sy) };
  }

  function draw(ctx) {
    for (const kind of Object.keys(KINDS)) {
      const k = KINDS[kind];
      st[kind].forEach((x, i) => {
        const c = screenCircle(x), sel = st.sel && st.sel.kind === kind && st.sel.i === i;
        ctx.globalAlpha = 0.16; ctx.fillStyle = k.color;
        ctx.beginPath(); ctx.arc(c.sx, c.sy, c.r, 0, 2 * Math.PI); ctx.fill();
        ctx.globalAlpha = sel ? 1 : 0.75; ctx.strokeStyle = sel ? "#facc15" : k.color;
        ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.beginPath(); ctx.arc(c.sx, c.sy, c.r, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = sel ? "#facc15" : k.color;                     // centre handle
        ctx.beginPath(); ctx.arc(c.sx, c.sy, 5, 0, 2 * Math.PI); ctx.fill();
      });
    }
    ctx.globalAlpha = 1;
  }

  // pick a sphere by its centre handle (nearest within 10 px)
  function pick(px, py) {
    let best = null, bd = 10;
    for (const kind of Object.keys(KINDS)) {
      st[kind].forEach((x, i) => {
        const [sx, sy] = project(...x.center), d = Math.hypot(sx - px, sy - py);
        if (d < bd) { bd = d; best = { kind, i }; }
      });
    }
    return best;
  }

  const local = (canvas, e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  return {
    st, load, add, draw, renderList,
    pointerDown(canvas, e) {
      const [px, py] = local(canvas, e), hit = pick(px, py);
      if (!hit) return false;
      st.sel = hit;
      const x = st[hit.kind][hit.i];
      st.drag = { ...hit, depth: project(...x.center)[2], moved: false };
      renderList();
      return true;
    },
    pointerMove(canvas, e) {
      if (!st.drag) return false;
      const [px, py] = local(canvas, e);
      const x = st[st.drag.kind][st.drag.i];
      x.center = unproject(px, py, st.drag.depth).map((v) => Number(v.toFixed(2)));
      st.drag.moved = true;
      return true;
    },
    async pointerUp() {
      const d = st.drag;
      st.drag = null;
      if (d && d.moved) { renderList(); await save(d.kind); }
    },
  };
}
