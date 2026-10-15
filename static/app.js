const $ = (id) => document.getElementById(id);
const chat = $("chat");

let tsChart = null;
let tierChart = null;

function tag(cls, text) {
  return `<span class="tag ${cls}">${text}</span>`;
}

function usd(v) {
  if (v == null) return "–";
  if (v === 0) return "$0.00";
  if (v < 0.001) return `$${(v * 1000).toFixed(3)}m`;
  return `$${v.toFixed(4)}`;
}

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function runQuery(q) {
  if (!q.trim()) return;
  const input = $("query-input");
  const form = $("query-form").querySelector("button");
  form.disabled = true;

  chat.insertAdjacentHTML("beforeend", `<div class="msg user">${escapeHtml(q)}</div>`);
  input.value = "";
  chat.scrollTop = chat.scrollHeight;

  try {
    const started = performance.now();
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    const uiLatency = Math.round(performance.now() - started);
    renderResponse(data, uiLatency);
    refreshStats();
  } catch (e) {
    chat.insertAdjacentHTML(
      "beforeend",
      `<div class="msg bot">⚠️ ${escapeHtml(e.message)}<div class="meta">${tag("cost", "check server logs / API key")}</div></div>`
    );
  } finally {
    form.disabled = false;
    input.focus();
  }
}

function renderResponse(data, uiLatency) {
  const { answer, route, usage, cache, context } = data;
  const isCache = route.tier === "cache";
  const tags = [];

  if (isCache) {
    tags.push(tag("cache", `cache hit · sim ${cache.similarity}`));
    tags.push(tag("cache", `${usage.total_ms}ms`));
  } else {
    tags.push(tag(route.tier, `${route.tier} → ${route.model}`));
    tags.push(tag("", `difficulty ${route.difficulty}`));
    tags.push(tag("cost", `${usage.cost_usd} · ${usage.total_ms}ms`));
    if (usage.escalated) tags.push(tag("large", "escalated"));
    if (context && context.length) tags.push(tag("", `${context.length} chunks retrieved`));
  }

  const reasons = (route.reasons || []).join(" · ");
  const html = `
    <div class="msg bot">
      ${escapeHtml(answer)}
      <div class="meta">${tags.join("")}</div>
      <div class="meta"><span class="tag" title="${escapeHtml(reasons)}">${escapeHtml(reasons.slice(0, 140))}</span></div>
      <div class="meta"><span class="tag">ui ${uiLatency}ms</span></div>
    </div>`;
  chat.insertAdjacentHTML("beforeend", html);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    const s = data.stats;
    const r = data.routing;
    const p = data.provider;

    $("provider-line").textContent =
      `llm: ${p.llm} · ` +
      `embeddings: ${p.embeddings} · small=${r.small_model} · large=${r.large_model} · threshold=${r.threshold}`;

    $("budget-badge").textContent = r.budget_usd_per_day
      ? `budget ${usd(r.spend_today_usd)} / ${usd(r.budget_usd_per_day)}${r.budget_exhausted ? " — EXHAUSTED" : ""}`
      : `spend today ${usd(r.spend_today_usd)} · no budget cap`;

    $("c-requests").textContent = s.total_requests;
    $("c-requests-sub").textContent = `${s.requests_last_24h} in last 24h · ${s.escalations} escalations`;
    $("c-hitrate").textContent = `${(s.cache_hit_rate * 100).toFixed(1)}%`;
    $("c-hits-sub").textContent = `${s.cache_hits} hits of ${s.total_requests}`;
    $("c-cost").textContent = usd(s.total_cost_usd);
    $("c-saved").textContent = `${usd(s.cost_saved_usd)} saved by cache · ${data.cache.entries} entries`;
    $("c-p95").textContent = `${Math.round(s.p95_latency_ms)}ms`;
    $("c-avg").textContent = `avg ${Math.round(s.avg_latency_ms)}ms`;

    drawTimeseries(s.series);
    drawTiers(s.by_tier);
    renderTable();
  } catch (e) {
    console.error("stats refresh failed", e);
  }
}

function drawTimeseries(series) {
  if (typeof Chart === "undefined") return;
  const labels = series.map((r) => r.bucket.slice(11));
  const costs = series.map((r) => +(r.cost || 0).toFixed(6));
  const lat = series.map((r) => Math.round(r.avg_ms || 0));

  const cfg = {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "cost ($)", data: costs, backgroundColor: "#f8514999", yAxisID: "y", borderRadius: 3 },
        { label: "avg latency (ms)", data: lat, type: "line", borderColor: "#58a6ff", backgroundColor: "#58a6ff", yAxisID: "y1", tension: 0.3, pointRadius: 2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8b949e", boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#8b949e", maxRotation: 0, autoSkip: true, font: { size: 10 } }, grid: { color: "#262d3855" } },
        y: { position: "left", ticks: { color: "#f85149", font: { size: 10 } }, grid: { color: "#262d3855" } },
        y1: { position: "right", ticks: { color: "#58a6ff", font: { size: 10 } }, grid: { drawOnChartArea: false } },
      },
    },
  };
  if (tsChart) { tsChart.destroy(); }
  tsChart = new Chart($("timeseries"), cfg);
}

function drawTiers(byTier) {
  if (typeof Chart === "undefined") return;
  const map = Object.fromEntries(byTier.map((t) => [t.route_tier, t.count]));
  const labels = ["cache", "small", "large"].filter((k) => map[k]);
  const data = labels.map((k) => map[k]);
  const colors = { cache: "#58a6ff", small: "#3fb950", large: "#d29922" };

  const cfg = {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: labels.map((l) => colors[l]), borderColor: "#161b22", borderWidth: 2 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: { legend: { position: "bottom", labels: { color: "#8b949e", boxWidth: 12, font: { size: 11 } } } },
    },
  };
  if (tierChart) tierChart.destroy();
  tierChart = new Chart($("tierchart"), cfg);
}

async function renderTable() {
  const res = await fetch("/api/requests?limit=40");
  const { requests } = await res.json();
  const tbody = $("log").querySelector("tbody");
  tbody.innerHTML = requests
    .map((r) => {
      const cacheHit = r.cache_hit === 1;
      const tier = cacheHit ? "cache" : r.route_tier || "–";
      const reasons = Array.isArray(r.reasons) ? r.reasons.join(" · ") : r.reasons || "";
      return `
        <tr class="${cacheHit ? "cache-row" : ""}">
          <td>${fmtTime(r.ts)}</td>
          <td class="q" title="${escapeHtml(reasons)}">${escapeHtml(r.query)}</td>
          <td><span class="pill ${tier}">${tier}${r.forced ? ` (${r.forced})` : ""}</span></td>
          <td>${escapeHtml(r.model || "–")}</td>
          <td class="diff">${r.difficulty != null ? (+r.difficulty).toFixed(3) : "–"}</td>
          <td>${cacheHit ? `hit ${(+r.cache_similarity).toFixed(3)}` : "miss"}</td>
          <td>${Math.round(r.total_ms || 0)}ms</td>
          <td>${(r.prompt_tokens || 0)}/${(r.completion_tokens || 0)}</td>
          <td>${usd(r.cost_usd)}</td>
        </tr>`;
    })
    .join("");
}

// --- wiring ----------------------------------------------------------------
$("query-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runQuery($("query-input").value);
});
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => runQuery(chip.dataset.q))
);
$("refresh").addEventListener("click", refreshStats);

refreshStats();
setInterval(refreshStats, 8000);
