// Shared network + safety helpers for every page (2-D workspace, 3-D viewer,
// explorer). A leaf module: no DOM, no app state — just fetch, job polling and
// HTML escaping, so the pages stop each carrying their own slightly different copy.

/** Escape text for interpolation into innerHTML (names typed by users, server
 *  messages). Anything user-controlled that goes into markup goes through this. */
export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** A readable message from a FastAPI error body: a string ``detail`` as is, a
 *  pydantic list as "field: message; …", anything else as JSON. */
export function errorText(detail, fallback = "request failed") {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const loc = (d.loc || []).filter((l) => l !== "body" && l !== "query").join(".");
      return loc ? `${loc}: ${d.msg}` : d.msg;
    }).join("; ");
  }
  return detail ? JSON.stringify(detail) : fallback;
}

/** fetch + JSON, throwing an ``Error`` with the server's message (and ``.status``)
 *  on a non-2xx response. A 204 returns null. */
export async function fetchJSON(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers["content-type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  if (r.status === 204) return null;
  const j = await r.json().catch(() => null);
  if (!r.ok) {
    const err = new Error(errorText(j && j.detail, `${method} ${path} failed (${r.status})`));
    err.status = r.status;
    throw err;
  }
  return j;
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

/** Wait for job ``jid`` to finish. Always resolves with ``state`` "done" or
 *  "error" — an ``interrupted`` job (the server restarted mid-run), a vanished job
 *  (404) or a server that stays unreachable all become "error" with a message, so
 *  no caller can spin forever or treat them as success. Brief outages (a restart
 *  in progress) are retried for ``maxFailures`` polls. */
export async function pollJob(jid, { interval = 400, maxFailures = 50 } = {}) {
  if (!jid) return { state: "error", error: "the server did not return a job id" };
  let failures = 0;
  for (;;) {
    let s;
    try {
      s = await fetchJSON("GET", `/jobs/${jid}`);
      failures = 0;
    } catch (e) {
      if (e.status === 404) return { state: "error", error: "the job is gone (server data reset?) — run it again" };
      if (++failures >= maxFailures) return { state: "error", error: `lost contact with the server: ${e.message}` };
      await sleep(interval);
      continue;
    }
    if (s.state === "done" || s.state === "error") return s;
    if (s.state === "interrupted") {
      return { ...s, state: "error", error: "the server restarted while this job ran — run it again" };
    }
    await sleep(interval);
  }
}

/** POST a job-starting request and wait for it (``{state: "done" | "error", …}``).
 *  A rejected request (422/404/409) resolves as an "error" job with its message. */
export async function submitJob(path, body = {}, opts) {
  let job;
  try {
    job = await fetchJSON("POST", path, body);
  } catch (e) {
    return { state: "error", error: e.message };
  }
  return pollJob(job && job.id, opts);
}

/** Run a long analysis (UQ, calibration, explorer optimise) as a background job and
 *  return its JSON result: POST ``path?background=true``, wait, then GET the job's
 *  result. A rejected request or a failed / interrupted job throws with its message.
 *  The request no longer has to stay open for the whole search. */
export async function runAnalysis(path, body, opts) {
  const sep = path.includes("?") ? "&" : "?";
  const st = await submitJob(`${path}${sep}background=true`, body, opts);
  if (st.state !== "done") throw new Error(st.error || "the analysis failed");
  return fetchJSON("GET", `/jobs/${st.id}/result`);
}
