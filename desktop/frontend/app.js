/* Kai v0.1 frontend — vanilla JS, no build step. */
const { invoke } = window.__TAURI__.core;

const el = (id) => document.getElementById(id);
const setState = (id, on, label) => {
  const e = el(id);
  e.textContent = label || (on ? "ON" : "OFF");
  e.className = on ? "on" : "off";
};

/* ---------- i18n (English / Português Brasil) ---------- */
const I18N = {
  en: {
    tagline: "Private AI computer. No cloud. No subscription. No data leaves.",
    nav_chat: "Chat", nav_history: "History", nav_settings: "Settings", nav_about: "About",
    status: "Status", refresh: "Refresh", chat: "Chat", send: "Send",
    history: "History", reload: "Reload", settings: "Settings", about: "About",
    theme: "Background", theme_black: "Black", theme_white: "White",
    language: "Language", about_text: "Kai runs its engine, MCP server and Ollama models entirely on your machine.",
    no_models: "no models — run: ollama pull qwen2.5:7b",
    ask_placeholder: "Ask your local models…", thinking: "thinking (kai)…",
    no_history: "No history yet — chat first, then reload.",
    msgs: "messages",
  },
  pt: {
    tagline: "Computador de IA privado. Sem nuvem. Sem assinatura. Nada sai da máquina.",
    nav_chat: "Conversa", nav_history: "Histórico", nav_settings: "Ajustes", nav_about: "Sobre",
    status: "Estado", refresh: "Atualizar", chat: "Conversa", send: "Enviar",
    history: "Histórico", reload: "Recarregar", settings: "Ajustes", about: "Sobre",
    theme: "Fundo", theme_black: "Preto", theme_white: "Branco",
    language: "Idioma", about_text: "O Kai roda seu motor, o servidor MCP e os modelos Ollama inteiramente na sua máquina.",
    no_models: "sem modelos — execute: ollama pull qwen2.5:7b",
    ask_placeholder: "Pergunte aos seus modelos locais…", thinking: "pensando (kai)…",
    no_history: "Sem histórico ainda — converse primeiro, depois recarregue.",
    msgs: "mensagens",
  },
};
let lang = localStorage.getItem("axiom-lang") || "en";
const t = (k) => (I18N[lang] && I18N[lang][k]) || I18N.en[k] || k;
function applyLang() {
  document.querySelectorAll("[data-i18n]").forEach((e) => {
    e.textContent = t(e.getAttribute("data-i18n"));
  });
  el("prompt").placeholder = t("ask_placeholder");
  document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
}

/* ---------- theme (black / white) ---------- */
function applyTheme() {
  const th = localStorage.getItem("axiom-theme") || "black";
  document.body.setAttribute("data-theme", th);
  el("sel-theme").value = th;
}

/* ---------- views + drawer ---------- */
function show(view) {
  ["chat", "history", "settings", "about"].forEach((v) => {
    el("view-" + v).classList.toggle("hidden", v !== view);
  });
  el("drawer").classList.add("hidden");
  if (view === "history") loadHistory();
}
el("btn-menu").addEventListener("click", () => el("drawer").classList.toggle("hidden"));
document.querySelectorAll("#drawer button").forEach((b) => {
  b.addEventListener("click", () => show(b.getAttribute("data-view")));
});

/* ---------- status ---------- */
async function refresh() {
  try {
    const s = await invoke("get_status");
    setState("st-ollama", s.ollama);
    setState("st-mcp", s.mcp);
    setState("st-bridge", s.bridge);
    el("st-models").textContent = s.models.length ? s.models.join(", ") : t("no_models");
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
    el("repo").textContent = "error: " + e;
  }
}
el("btn-mcp").addEventListener("click", async () => {
  const running = el("btn-mcp").textContent.startsWith("Stop");
  try { el("repo").textContent = await invoke(running ? "stop_mcp" : "start_mcp"); }
  catch (e) { el("repo").textContent = "error: " + e; }
  refresh();
});
el("btn-bridge").addEventListener("click", async () => {
  const running = el("btn-bridge").textContent.startsWith("Stop");
  try { el("repo").textContent = await invoke(running ? "stop_bridge" : "start_bridge"); }
  catch (e) { el("repo").textContent = "error: " + e; }
  refresh();
});
el("btn-refresh").addEventListener("click", refresh);

/* ---------- live chat (sequential Q&A, footer metadata) ---------- */
function fmtTime(d) {
  return d.toLocaleString(lang === "pt" ? "pt-BR" : "en-US");
}
function addMsg(role, text, meta) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  wrap.appendChild(body);
  if (meta) {
    const foot = document.createElement("div");
    foot.className = "msg-meta";
    foot.textContent = meta;
    wrap.appendChild(foot);
  }
  el("live").appendChild(wrap);
  wrap.scrollIntoView({ block: "nearest" });
  return body;
}
async function send() {
  const model = el("model").value;
  const prompt = el("prompt").value.trim();
  if (!model || !prompt) return;
  el("btn-send").disabled = true;  addMsg("user", prompt, (lang === "pt" ? "você" : "you") + " · " + fmtTime(new Date()));
  el("prompt").value = "";
  const t0 = performance.now();
  // Footer div created upfront so the stamp has a home even if the
  // model streams nothing before done.
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const bodyEl = document.createElement("div");
  bodyEl.className = "msg-body";
  const footEl = document.createElement("div");
  footEl.className = "msg-meta";
  footEl.textContent = "…";
  wrap.appendChild(bodyEl);
  wrap.appendChild(footEl);
  el("live").appendChild(wrap);
  wrap.scrollIntoView({ block: "nearest" });
  const { listen } = window.__TAURI__.event;
  const secs = () => ((performance.now() - t0) / 1000).toFixed(1) + "s";
  const unlistenToken = await listen("chat-token", (e) => {
    bodyEl.textContent += e.payload;
    bodyEl.scrollTop = bodyEl.scrollHeight;
  });
  const unlistenDone = await listen("chat-done", (e) => {
    if (e.payload) bodyEl.textContent += "\n[error: " + e.payload + "]";
    if (!bodyEl.textContent) bodyEl.textContent = "—";
    footEl.textContent = "kai · " + model + " · " + secs() + " · " + fmtTime(new Date());
    el("btn-send").disabled = false;
    unlistenToken();
    unlistenDone();
  });
  try {
    await invoke("chat_stream", { model, prompt });
  } catch (e) {
    bodyEl.textContent = "error: " + e;
    el("btn-send").disabled = false;
  }
}
el("btn-send").addEventListener("click", send);
el("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

/* ---------- history (date sessions, sequential, footers) ---------- */
let histCache = [];
function dayOf(ts) {
  try { return new Date(ts).toISOString().slice(0, 10); } catch (e) { return "?"; }
}
async function loadHistory() {
  const box = el("sessions");
  const msgs = el("messages");
  box.innerHTML = ""; msgs.innerHTML = "";
  let rows;
  try {
    const r = await invoke("history");
    rows = r.messages || [];
  } catch (e) {
    box.textContent = "error: " + e;
    return;
  }
  histCache = rows;
  if (!rows.length) { box.textContent = t("no_history"); return; }
  const groups = {};
  rows.forEach((m, i) => {
    const d = dayOf(m.ts);
    (groups[d] = groups[d] || []).push({ ...m, _i: i });
  });
  Object.keys(groups).sort().reverse().forEach((d) => {
    const b = document.createElement("button");
    b.className = "sess";
    b.textContent = d + " — " + groups[d].length + " " + t("msgs");
    b.addEventListener("click", () => showSession(d, groups[d]));
    box.appendChild(b);
  });
}
function showSession(day, items) {
  const msgs = el("messages");
  msgs.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = day;
  msgs.appendChild(h);
  items.forEach((m) => {
    const role = m.role === "user" ? "user" : "assistant";
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = m.content || "";
    wrap.appendChild(body);
    const foot = document.createElement("div");
    foot.className = "msg-meta";
    const when = m.ts ? fmtTime(new Date(m.ts)) : "";
    foot.textContent = role === "user"
      ? ((lang === "pt" ? "você" : "you") + (when ? " · " + when : ""))
      : ("kai" + (when ? " · " + when : ""));
    wrap.appendChild(foot);
    msgs.appendChild(wrap);
  });
}
el("btn-reload-history").addEventListener("click", loadHistory);

/* ---------- settings ---------- */
el("sel-theme").addEventListener("change", (e) => {
  localStorage.setItem("axiom-theme", e.target.value);
  applyTheme();
});
el("sel-lang").addEventListener("change", (e) => {
  lang = e.target.value;
  localStorage.setItem("axiom-lang", lang);
  applyLang();
  refresh();
});

/* ---------- boot ---------- */
el("sel-lang").value = lang;
applyLang();
applyTheme();
refresh();
setInterval(refresh, 15000);
