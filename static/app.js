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
    // preventScroll: focusing the input must not yank the page around
    // (the console sits below the fold on narrow panels).
    input.focus({ preventScroll: true });
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
    tags.push(tag("cost", `${usd(usage.cost_usd)} · ${usage.total_ms}ms`));
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

    const offline = p.llm.toLowerCase().includes("mock");
    const badge = offline
      ? '<span class="mock">offline demo</span>'
      : '<span class="live">live</span>';
    $("provider-line").innerHTML =
      `${badge} llm: ${escapeHtml(p.llm)} · ` +
      `embeddings: ${escapeHtml(p.embeddings)} · small=${escapeHtml(r.small_model)} · ` +
      `large=${escapeHtml(r.large_model)} · threshold=${r.threshold}`;

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

  // The API only returns minute-buckets that actually had traffic, so sparse
  // traffic used to render as a couple of isolated giant bars floating in an
  // empty plot. Rebuild a continuous timeline instead: every slot in the
  // data span is present (cost 0 / latency null where nothing happened), so
  // the chart reads like a real time series.
  const MIN = 60000;
  const labels = [];
  const costs = [];
  const lat = [];

  if (series.length) {
    const byMin = new Map();
    for (const r of series) {
      // Buckets come from SQLite strftime(..., 'unixepoch') -> UTC minutes.
      const t = Date.parse(`${r.bucket.replace(" ", "T")}:00Z`);
      if (Number.isNaN(t)) continue;
      const slot = byMin.get(t) || { cost: 0, lats: [] };
      slot.cost += +(r.cost || 0);
      if (+r.avg_ms > 0) slot.lats.push(+r.avg_ms);
      byMin.set(t, slot);
    }
    const times = [...byMin.keys()].sort((a, b) => a - b);
    if (times.length) {
      let start = Math.floor(times[0] / MIN) * MIN;
      let end = Math.ceil(times[times.length - 1] / MIN) * MIN;
      // A single-minute window stays readable instead of one naked point.
      if (end - start < 15 * MIN) { start -= 7 * MIN; end += 7 * MIN; }
      // Cap resolution so a full window can't flood the canvas.
      const step = Math.max(MIN, Math.ceil((end - start) / MIN / 180) * MIN);
      for (let t = start; t <= end; t += step) {
        let cost = 0;
        const lats = [];
        let seen = false;
        for (let m = t; m < t + step; m += MIN) {
          const slot = byMin.get(m);
          if (slot) {
            seen = true;
            cost += slot.cost;
            lats.push(...slot.lats);
          }
        }
        labels.push(new Date(t).toISOString().slice(11, 16));
        costs.push(seen ? +cost.toFixed(6) : 0);
        lat.push(lats.length ? Math.round(lats.reduce((a, b) => a + b, 0) / lats.length) : null);
      }
    }
  }

  const emptyState = {
    id: "emptyState",
    afterDraw(chart) {
      if (labels.length) return;
      const { ctx, chartArea } = chart;
      if (!chartArea) return;
      ctx.save();
      ctx.fillStyle = "#96a2c4";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("no traffic in this window yet",
        (chartArea.left + chartArea.right) / 2,
        (chartArea.top + chartArea.bottom) / 2);
      ctx.restore();
    },
  };

  const cfg = {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "cost ($)",
          data: costs,
          borderColor: "#8b5cf6",
          backgroundColor: "rgba(139, 92, 246, 0.16)",
          fill: true,
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 0,
          pointHitRadius: 10,
          spanGaps: true,
          yAxisID: "y",
        },
        {
          label: "avg latency (ms)",
          data: lat,
          borderColor: "#22d3ee",
          backgroundColor: "#22d3ee",
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 2.5,
          pointHoverRadius: 5,
          pointHitRadius: 10,
          spanGaps: true,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#96a2c4", boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: "#96a2c4", maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } }, grid: { color: "rgba(148, 163, 255, 0.07)" } },
        y: { position: "left", beginAtZero: true, ticks: { color: "#a78bfa", font: { size: 10 }, callback: (v) => usd(v) }, grid: { color: "rgba(148, 163, 255, 0.07)" } },
        y1: { position: "right", beginAtZero: true, ticks: { color: "#22d3ee", font: { size: 10 }, callback: (v) => `${v}ms` }, grid: { drawOnChartArea: false } },
      },
    },
    plugins: [emptyState],
  };
  if (tsChart) { tsChart.destroy(); }
  tsChart = new Chart($("timeseries"), cfg);
}

function drawTiers(byTier) {
  if (typeof Chart === "undefined") return;
  const map = Object.fromEntries(byTier.map((t) => [t.route_tier, t.count]));
  const labels = ["cache", "small", "large"].filter((k) => map[k]);
  const data = labels.map((k) => map[k]);
  const colors = { cache: "#22d3ee", small: "#34d399", large: "#fbbf24" };

  const cfg = {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: labels.map((l) => colors[l]), borderColor: "rgba(19, 23, 42, 0.9)", borderWidth: 2, hoverOffset: 6 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "64%",
      plugins: { legend: { position: "bottom", labels: { color: "#96a2c4", boxWidth: 12, font: { size: 11 } } } },
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
