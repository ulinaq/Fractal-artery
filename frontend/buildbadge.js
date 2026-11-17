// Build badge: shows the frontend build hash and flags when this tab is stale —
// i.e. the app's .html/.js/.css changed on the server since the page loaded — so
// the user knows a reload is needed (no more silently running old JavaScript).
const badge = document.getElementById("build-badge");
if (badge) {
  let loaded = null;

  async function currentBuild() {
    try { return (await (await fetch("/frontend-version")).json()).build; }
    catch { return null; }
  }

  function markStale() {
    badge.textContent = "⚠ new build — reload";
    badge.classList.add("stale");
    badge.title = "the app was updated on the server since this tab opened — click to reload";
    badge.onclick = () => location.reload();
  }

  async function check() {
    const b = await currentBuild();
    if (!b) return;
    if (loaded === null) { loaded = b; badge.textContent = "build " + b; }
    else if (b !== loaded && !badge.classList.contains("stale")) markStale();
  }

  check();
  setInterval(check, 20000);                       // poll periodically…
  document.addEventListener("visibilitychange", () => { if (!document.hidden) check(); });  // …and on refocus
}
