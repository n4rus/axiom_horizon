/* Axiom Local v0.1 frontend — vanilla JS, no build step. */
const { invoke } = window.__TAURI__.core;

const el = (id) => document.getElementById(id);
const setState = (id, on, label) => {
  const e = el(id);
  e.textContent = label || (on ? "ON" : "OFF");
  e.className = on ? "on" : "off";
};

async function refresh() {
  try {
    const s = await invoke("get_status");
    setState("st-ollama", s.ollama);
    setState("st-mcp", s.mcp);
    setState("st-bridge", s.bridge);
    el("st-models").textContent = s.models.length ? s.models.join(", ") : "no models — run: ollama pull qwen2.5:7b";
    el("repo").textContent = s.repo ? "repo: " + s.repo : "";
    const sel = el("model");
    const cur = sel.value;
    sel.innerHTML = "";
    s.models.forEach((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    });
    if (s.models.includes(cur)) sel.value = cur;
    el("btn-mcp").textContent = s.mcp ? "Stop MCP" : "Start MCP";
    el("btn-bridge").textContent = s.bridge ? "Stop Bridge" : "Start Bridge";
  } catch (e) {
    el("repo").textContent = "backend error: " + e;
  }
}

async function toggle(cmd, on, refreshAfter) {
  try {
    const msg = await invoke(cmd);
    el("repo").textContent = msg;
  } catch (e) {
    el("repo").textContent = "error: " + e;
  }
  refreshAfter();
}

el("btn-mcp").addEventListener("click", async () => {
  const running = el("btn-mcp").textContent.startsWith("Stop");
  await toggle(running ? "stop_mcp" : "start_mcp", running, refresh);
});
el("btn-bridge").addEventListener("click", async () => {
  const running = el("btn-bridge").textContent.startsWith("Stop");
  await toggle(running ? "stop_bridge" : "start_bridge", running, refresh);
});
el("btn-refresh").addEventListener("click", refresh);

async function send() {
  const model = el("model").value;
  const prompt = el("prompt").value.trim();
  if (!model || !prompt) return;
  el("btn-send").disabled = true;
  el("out").textContent = "thinking (local)…";
  try {
    el("out").textContent = await invoke("chat", { model, prompt });
  } catch (e) {
    el("out").textContent = "error: " + e;
  }
  el("btn-send").disabled = false;
}
el("btn-send").addEventListener("click", send);
el("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

refresh();
setInterval(refresh, 15000);
