// perfusion — hand-editing the 3-D trees of a 3-D project in the 3-D viewer.
//
// The trees are edited as networks (nodes + segments, each segment with its own
// radius), like the 2-D Vessels editor:
//   click a node / vessel ........ select it (a vessel's radius is then editable)
//   drag a node .................. move it in the screen plane (its depth is kept —
//                                  rotate the view to move it along another axis)
//   Shift+click empty space ...... grow a new branch from the selected node
//   S / Split .................... split the selected vessel at its midpoint
//   I / Toggle inlet ............. make the selected node an artery inlet / vein outlet
//   Del / Delete ................. delete the selected node or vessel
//   ⌘/Ctrl+Z ..................... undo
// Every edit is saved with PUT /projects/{id}/editable (which rebuilds the tree and
// invalidates the stale result); the server's reindexed tree is then reloaded.

const KINDS = ["artery", "vein"];
const COLORS = { artery: "#ff6b6b", vein: "#5b8dff" };
const NODE_PX = 9, SEG_PX = 6, DRAG_PX = 3;

const clone = (m) => JSON.parse(JSON.stringify(m));

/**
 * @param {object} o
 * @param {HTMLCanvasElement} o.canvas
 * @param {object} o.vp                     shared viewport (eventXY)
 * @param {string} o.projectId
 * @param {(x:number,y:number,z:number)=>number[]} o.project   world → [sx, sy, depth]
 * @param {(sx:number,sy:number,depth:number)=>number[]} o.unproject  screen+depth → world
 * @param {()=>Promise<void>} o.onSaved     re-show the (now unsolved) project
 * @param {(msg:string, err?:boolean)=>void} o.onInfo
 */
export function createEditor3d(o) {
  const ed = {
    active: false,
    model: { artery: null, vein: null },
    sel: null,               // {kind, type: "node"|"seg", i}
    drag: null,              // {kind, i, depth, x0, y0, moved}
    history: [],
    busy: false,
  };

  async function fetchModel() {
    const r = await fetch(`/projects/${o.projectId}/editable`);
    if (!r.ok) throw new Error("could not load the trees for editing");
    const m = await r.json();
    ed.model = { artery: m.artery, vein: m.vein };
  }

  ed.setActive = async (on) => {
    ed.active = on;
    ed.sel = null; ed.drag = null;
    if (on) await fetchModel();
    ed.refreshPanel();
  };

  // ── geometry helpers ────────────────────────────────────────────────────────
  const pos = (net, i) => [net.node_x[i], net.node_y[i], net.node_z[i]];
  const screen = (net, i) => o.project(...pos(net, i));

  function hit(px, py) {
    let best = null;
    for (const kind of KINDS) {                       // nodes first: they sit on vessels
      const net = ed.model[kind];
      if (!net) continue;
      for (let i = 0; i < net.node_x.length; i++) {
        const [sx, sy] = screen(net, i);
        const d = Math.hypot(sx - px, sy - py);
        if (d <= NODE_PX && (!best || d < best.d)) best = { kind, type: "node", i, d };
      }
    }
    if (best) return best;
    for (const kind of KINDS) {
      const net = ed.model[kind];
      if (!net) continue;
      for (let s = 0; s < net.seg_start.length; s++) {
        const [ax, ay] = screen(net, net.seg_start[s]);
        const [bx, by] = screen(net, net.seg_end[s]);
        const vx = bx - ax, vy = by - ay, L2 = vx * vx + vy * vy || 1;
        const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / L2));
        const d = Math.hypot(ax + t * vx - px, ay + t * vy - py);
        if (d <= SEG_PX && (!best || d < best.d)) best = { kind, type: "seg", i: s, d };
      }
    }
    return best;
  }

  /** A node's canvas position (CSS px) — for tests and tooling. */
  ed.screenOf = (kind, i) => screen(ed.model[kind], i).slice(0, 2);

  const inlets = (net) => (net.inlet_nodes && net.inlet_nodes.length ? net.inlet_nodes : [net.root_node]);

  // ── saving ──────────────────────────────────────────────────────────────────
  // save one tree; ``before`` is the undo snapshot (default: the model now)
  async function commit(kind, next, message, before = clone(ed.model)) {
    if (ed.busy) return false;
    ed.busy = true;
    // remember the selection by position: the server reindexes nodes and segments
    const sel = ed.sel && ed.sel.kind === kind && next ? ed.sel : null;
    const selPos = sel && sel.type === "node" ? pos(next, sel.i) : null;
    const segPos = sel && sel.type === "seg" && sel.i < next.seg_start.length
      ? [pos(next, next.seg_start[sel.i]), pos(next, next.seg_end[sel.i])] : null;
    const body = {};
    body[kind] = next ? {
      node_x: next.node_x, node_y: next.node_y, node_z: next.node_z,
      seg_start: next.seg_start, seg_end: next.seg_end, seg_radius: next.seg_radius,
      inlet_nodes: next.inlet_nodes, root_node: next.root_node,
    } : { node_x: [], node_y: [], node_z: [], seg_start: [], seg_end: [], seg_radius: [] };
    try {
      const r = await fetch(`/projects/${o.projectId}/editable`, {
        method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        const why = typeof j.detail === "string" ? j.detail
          : Array.isArray(j.detail) ? j.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ")
          : `status ${r.status}`;
        throw new Error(`edit failed — ${why}`);
      }
      ed.history.push(before);
      if (ed.history.length > 50) ed.history.shift();
      await fetchModel();
      if (selPos) ed.sel = nearestNode(kind, selPos);
      else if (segPos) ed.sel = segmentBetween(kind, segPos);
      else if (sel) ed.sel = null;
      o.onInfo("saving…");
      await o.onSaved();
      o.onInfo(`${message} · re-run to solve the edited trees`);
      return true;
    } catch (e) {
      ed.model = before;
      ed.sel = null;                             // the selection may not exist any more
      o.onInfo(e.message, true);
      return false;
    } finally {
      ed.busy = false;
      ed.refreshPanel();
    }
  }

  function nearestNode(kind, p) {
    const net = ed.model[kind];
    if (!net) return null;
    let best = -1, bd = Infinity;
    for (let i = 0; i < net.node_x.length; i++) {
      const d = (net.node_x[i] - p[0]) ** 2 + (net.node_y[i] - p[1]) ** 2 + (net.node_z[i] - p[2]) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? { kind, type: "node", i: best } : null;
  }

  function segmentBetween(kind, [p, q]) {
    const a = nearestNode(kind, p), b = nearestNode(kind, q), net = ed.model[kind];
    if (!a || !b) return null;
    const s = net.seg_start.findIndex((u, k) => (u === a.i && net.seg_end[k] === b.i)
                                               || (u === b.i && net.seg_end[k] === a.i));
    return s >= 0 ? { kind, type: "seg", i: s } : null;
  }

  ed.undo = async () => {
    const prev = ed.history.pop();
    if (!prev || ed.busy) return;
    const changed = KINDS.filter((k) => JSON.stringify(prev[k]) !== JSON.stringify(ed.model[k]));
    ed.sel = null;
    for (const k of changed) {
      if (await commit(k, prev[k], "undone")) ed.history.pop();   // an undo isn't undoable
    }
  };

  // ── edits ───────────────────────────────────────────────────────────────────
  function incidentRadius(net, i) {
    const r = [];
    net.seg_start.forEach((a, s) => { if (a === i || net.seg_end[s] === i) r.push(net.seg_radius[s]); });
    return r.length ? Math.min(...r) : 1;
  }

  ed.addBranch = (px, py) => {
    if (!ed.sel || ed.sel.type !== "node") return false;
    const { kind, i } = ed.sel;
    const net = clone(ed.model[kind]);
    const [, , depth] = screen(net, i);
    const [x, y, z] = o.unproject(px, py, depth);
    const radius = Number((0.8 * incidentRadius(net, i)).toPrecision(3));   // before the push
    net.node_x.push(x); net.node_y.push(y); net.node_z.push(z);
    const j = net.node_x.length - 1;
    net.seg_start.push(i); net.seg_end.push(j); net.seg_radius.push(radius);
    ed.sel = { kind, type: "node", i: j };
    commit(kind, net, `${kind}: branch added`);
    return true;
  };

  ed.split = () => {
    if (!ed.sel || ed.sel.type !== "seg") return;
    const { kind, i: s } = ed.sel;
    const net = clone(ed.model[kind]);
    const a = net.seg_start[s], b = net.seg_end[s];
    net.node_x.push((net.node_x[a] + net.node_x[b]) / 2);
    net.node_y.push((net.node_y[a] + net.node_y[b]) / 2);
    net.node_z.push((net.node_z[a] + net.node_z[b]) / 2);
    const m = net.node_x.length - 1;
    net.seg_end[s] = m;
    net.seg_start.push(m); net.seg_end.push(b); net.seg_radius.push(net.seg_radius[s]);
    ed.sel = { kind, type: "node", i: m };
    commit(kind, net, `${kind}: vessel split`);
  };

  ed.toggleInlet = () => {
    if (!ed.sel || ed.sel.type !== "node") return;
    const { kind, i } = ed.sel;
    const net = clone(ed.model[kind]);
    const cur = new Set(inlets(net));
    if (cur.has(i)) {
      if (cur.size === 1) {
        o.onInfo(`the ${kind} needs at least one ${kind === "artery" ? "inlet" : "outlet"}`, true);
        return;
      }
      cur.delete(i);
    } else cur.add(i);
    net.inlet_nodes = [...cur].sort((p, q) => p - q);
    if (!cur.has(net.root_node)) net.root_node = net.inlet_nodes[0];
    commit(kind, net, `${kind}: ${kind === "artery" ? "inlet" : "outlet"} ${cur.has(i) ? "set" : "unset"}`);
  };

  ed.remove = () => {
    if (!ed.sel) return;
    const { kind, type, i } = ed.sel;
    const net = clone(ed.model[kind]);
    const keep = net.seg_start.map((a, s) => (type === "seg" ? s !== i
      : a !== i && net.seg_end[s] !== i));
    for (const key of ["seg_start", "seg_end", "seg_radius"]) net[key] = net[key].filter((_, s) => keep[s]);
    if (type === "node") {
      const rest = inlets(net).filter((n) => n !== i);
      net.inlet_nodes = rest;
      if (net.root_node === i) net.root_node = rest.length ? rest[0] : null;
    }
    ed.sel = null;
    if (!net.seg_start.length) {
      o.onInfo(`that would delete the whole ${kind} tree`, true);
      return;
    }
    commit(kind, net, `${kind}: ${type === "seg" ? "vessel" : "node"} deleted`);
  };

  ed.setRadius = (r) => {
    if (!ed.sel || ed.sel.type !== "seg" || !(r > 0)) return;
    const { kind, i } = ed.sel;
    const net = clone(ed.model[kind]);
    net.seg_radius[i] = r;
    commit(kind, net, `${kind}: radius ${r}`);
  };

  // ── pointer: returns true when the editor consumed the event ────────────────
  ed.pointerDown = (e) => {
    if (!ed.active || ed.busy) return false;
    const [px, py] = o.vp.eventXY(e);
    if (e.shiftKey && ed.addBranch(px, py)) return true;
    const h = hit(px, py);
    if (!h) { ed.sel = null; ed.refreshPanel(); return false; }   // empty: rotate the view
    ed.sel = { kind: h.kind, type: h.type, i: h.i };
    if (h.type === "node") {
      const net = ed.model[h.kind];
      ed.drag = { kind: h.kind, i: h.i, depth: screen(net, h.i)[2], x0: px, y0: py,
                  moved: false, orig: pos(net, h.i) };
    }
    ed.refreshPanel();
    return true;
  };

  ed.pointerMove = (e) => {
    const d = ed.drag;
    if (!d) return false;
    const [px, py] = o.vp.eventXY(e);
    if (!d.moved && Math.hypot(px - d.x0, py - d.y0) < DRAG_PX) return true;
    d.moved = true;
    const [x, y, z] = o.unproject(px, py, d.depth);
    const net = ed.model[d.kind];
    net.node_x[d.i] = x; net.node_y[d.i] = y; net.node_z[d.i] = z;
    return true;
  };

  ed.pointerUp = () => {
    const d = ed.drag;
    ed.drag = null;
    if (!d || !d.moved) return;
    const before = clone(ed.model);                       // undo restores the pre-drag spot
    [before[d.kind].node_x[d.i], before[d.kind].node_y[d.i], before[d.kind].node_z[d.i]] = d.orig;
    commit(d.kind, clone(ed.model[d.kind]), `${d.kind}: node moved`, before);
  };

  ed.keyDown = (e) => {
    if (!ed.active || e.target.closest("input, select, textarea")) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); ed.undo(); return; }
    if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); ed.remove(); }
    else if (e.key.toLowerCase() === "i") ed.toggleInlet();
    else if (e.key.toLowerCase() === "s") ed.split();
    else if (e.key === "Escape") { ed.sel = null; ed.refreshPanel(); }
  };

  // ── drawing (replaces the plain tree rendering while editing) ───────────────
  ed.draw = (ctx, zs) => {
    for (const kind of KINDS) {
      const net = ed.model[kind];
      if (!net) continue;
      ctx.lineCap = "round";
      for (let s = 0; s < net.seg_start.length; s++) {
        const [ax, ay] = screen(net, net.seg_start[s]);
        const [bx, by] = screen(net, net.seg_end[s]);
        const on = ed.sel && ed.sel.kind === kind && ed.sel.type === "seg" && ed.sel.i === s;
        ctx.strokeStyle = on ? "#facc15" : COLORS[kind];
        ctx.lineWidth = Math.max(on ? 3 : 1, net.seg_radius[s] * 0.9 * zs);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
      const ins = new Set(inlets(net));
      for (let i = 0; i < net.node_x.length; i++) {
        const [sx, sy] = screen(net, i);
        const on = ed.sel && ed.sel.kind === kind && ed.sel.type === "node" && ed.sel.i === i;
        ctx.fillStyle = on ? "#facc15" : ins.has(i) ? "#ffffff" : COLORS[kind];
        const r = on ? 5 : ins.has(i) ? 4.5 : 2.6;
        ctx.beginPath(); ctx.arc(sx, sy, r, 0, 2 * Math.PI); ctx.fill();
        if (ins.has(i)) { ctx.strokeStyle = COLORS[kind]; ctx.lineWidth = 2; ctx.stroke(); }
      }
    }
  };

  // ── side panel ──────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  ed.refreshPanel = () => {
    $("v3-edit-panel").hidden = !ed.active;
    const s = ed.sel;
    const net = s && ed.model[s.kind];
    let text = "Nothing selected — click a node or vessel.";
    if (s && net && s.type === "node") {
      const isIn = inlets(net).includes(s.i);
      const word = s.kind === "artery" ? "inlet" : "outlet";
      const p = pos(net, s.i).map((v) => v.toFixed(1)).join(", ");
      text = `${s.kind} node ${s.i} (${p})${isIn ? ` · ${word}` : ""} — drag to move, `
        + "Shift+click to branch";
      $("v3-inlet").textContent = isIn ? `Unset ${word}` : `Make ${word}`;
    } else if (s && net && s.type === "seg") {
      text = `${s.kind} vessel ${s.i}`;
      $("v3-rad").value = Number(net.seg_radius[s.i].toPrecision(4));
    }
    $("v3-sel").textContent = text;
    $("v3-rad-row").hidden = !(s && s.type === "seg");
    $("v3-split").hidden = !(s && s.type === "seg");
    $("v3-occlude").hidden = !(s && s.type === "seg");
    $("v3-inlet").hidden = !(s && s.type === "node");
    $("v3-del").hidden = !s;
    $("v3-undo").disabled = !ed.history.length;
  };

  $("v3-rad").addEventListener("change", (e) => ed.setRadius(Number(e.target.value)));
  $("v3-split").addEventListener("click", () => ed.split());
  $("v3-occlude").addEventListener("click", () => {     // an embolus: radius × 0.001
    if (ed.sel && ed.sel.type === "seg") {
      ed.setRadius(Math.max(1e-6, ed.model[ed.sel.kind].seg_radius[ed.sel.i] * 1e-3));
    }
  });
  $("v3-inlet").addEventListener("click", () => ed.toggleInlet());
  $("v3-del").addEventListener("click", () => ed.remove());
  $("v3-undo").addEventListener("click", () => ed.undo());
  window.addEventListener("keydown", (e) => ed.keyDown(e));
  return ed;
}
