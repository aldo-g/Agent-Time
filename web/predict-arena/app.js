const summaryContainer = document.getElementById("summary-metrics");
const agentGrid = document.getElementById("agent-grid");
const detailTitle = document.getElementById("detail-title");
const detailSubtitle = document.getElementById("detail-subtitle");
const detailContainer = document.getElementById("agent-details");

let agents = [];
let activeSlug = null;

const currency = (value) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

const percent = (value) => `${(value * 100).toFixed(0)}%`;
const formatShares = (value) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(Number(value || 0));
const formatProb = (value) => (value == null ? "n/a" : `${(value * 100).toFixed(1)}%`);
const formatMana = (value) =>
  value == null ? "n/a" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
const formatSignedMana = (value) => {
  if (value == null) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${formatMana(Math.abs(value))}`;
};
const formatDelta = (value) => {
  if (value == null) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${(Math.abs(value) * 100).toFixed(1)}pp`;
};

function renderSummary(summary, lastUpdated) {
  const entries = [
    { label: "Active Markets", value: summary.activeMarkets },
    { label: "Mana In Play", value: currency(summary.manaInPlay) },
    { label: "Trades Today", value: summary.totalTradesToday },
    { label: "Last Update", value: new Date(lastUpdated).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }
  ];
  summaryContainer.innerHTML = entries
    .map(
      ({ label, value }) => `
      <article class="metric-card">
        <p class="metric-label">${label}</p>
        <p class="metric-value">${value}</p>
      </article>`
    )
    .join("");
}

function renderAgentCard(agent) {
  const {
    slug,
    name,
    provider,
    model,
    cash,
    dailyPnl,
    winRate,
    openPositions,
    totalAssets,
    color,
    colorMuted
  } = agent;
  const pnlSign = dailyPnl >= 0 ? "+" : "";
  const selected = slug === activeSlug ? "active" : "";
  const accentStyle = `style="--agent-accent-strong: ${color || "var(--accent)"}; --agent-accent-soft: ${
    colorMuted || "var(--accent-muted)"
  };"`;
  return `
    <article class="agent-card ${selected}" data-slug="${slug}" ${accentStyle}>
      <header>
        <div>
          <p class="agent-name">${name}</p>
          <p class="badge">${provider}</p>
        </div>
        <small class="muted">${model}</small>
      </header>
      <div class="stat-row"><span>Cash</span><strong>${currency(cash)}</strong></div>
      <div class="stat-row"><span>Total Assets</span><strong>${currency(totalAssets || 0)}</strong></div>
      <div class="stat-row"><span>Daily PnL</span><strong>${pnlSign}${currency(Math.abs(dailyPnl))}</strong></div>
      <div class="stat-row"><span>Win Rate</span><strong>${percent(winRate)}</strong></div>
      <div class="stat-row"><span>Open Positions</span><strong>${openPositions}</strong></div>
    </article>
  `;
}

function renderAgentGrid() {
  agentGrid.innerHTML = agents.map(renderAgentCard).join("");
  agentGrid.querySelectorAll(".agent-card").forEach((card) => {
    card.addEventListener("click", () => {
      const slug = card.dataset.slug;
      selectAgent(slug);
    });
  });
}

function tradeStatusPill(trade) {
  if (trade.status === "OPEN") {
    return `<span class="pill open">Open</span>`;
  }
  const win = (trade.settlement ?? 0) >= 0;
  return `<span class="pill ${win ? "resolved" : "loss"}">${win ? "Win" : "Loss"}</span>`;
}

function renderTradesTable(trades = []) {
  if (!trades.length) {
    return "<p class='empty-state'>No trades placed yet.</p>";
  }
  return `
    <div class="table-wrapper">
      <table class="trades-table">
        <thead>
          <tr>
            <th>When</th>
            <th>Market</th>
            <th>Action</th>
            <th>Stake</th>
            <th>Prob Δ</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${trades
            .map((trade) => {
              const delta = trade.probAfter && trade.probBefore ? trade.probAfter - trade.probBefore : 0;
              return `
                <tr>
                  <td>${new Date(trade.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
                  <td><a href="${trade.marketUrl}" target="_blank" rel="noreferrer">${trade.market}</a></td>
                  <td>${trade.action} ${trade.outcome}</td>
                  <td>${currency(trade.amount)}</td>
                  <td>${(delta * 100).toFixed(1)}pp</td>
                  <td>${tradeStatusPill(trade)}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderPositions(positions = []) {
  if (!positions.length) {
    return "<p class='empty-state'>No open positions found.</p>";
  }
  const summary = positions.reduce(
    (acc, position) => {
      const shares = Number(position.shares || 0);
      const avg = Number(position.avg_price ?? position.avgPrice);
      const mark = Number(position.mark_price ?? position.markPrice);
      if (Number.isFinite(shares) && Number.isFinite(mark)) {
        acc.mana += shares * mark;
      }
      const pnl = Number(position.pnl);
      if (Number.isFinite(pnl)) {
        acc.pnl += pnl;
      } else if (Number.isFinite(shares) && Number.isFinite(avg) && Number.isFinite(mark)) {
        acc.pnl += (mark - avg) * shares;
      }
      return acc;
    },
    { mana: 0, pnl: 0 }
  );
  const pnlClass = summary.pnl >= 0 ? "positive" : "negative";
  return `
    <div class="positions-summary">
      <span>${positions.length} markets</span>
      <span>Mana at work: ${formatMana(summary.mana)}</span>
      <span class="${pnlClass}">Unrealized PnL: ${formatSignedMana(summary.pnl)}</span>
    </div>
    <div class="positions-head">
      <span>Market</span>
      <span>Bought</span>
      <span>Current</span>
      <span>Win Value</span>
      <span>Δ Prob</span>
      <span>PnL</span>
    </div>
    <ul class="positions-list">
      ${positions
        .map(
          (position) => {
            const shares = Number(position.shares || 0);
            const avg = Number(position.avg_price ?? position.avgPrice);
            const mark = Number(position.mark_price ?? position.markPrice);
            const entryProb = Number.isFinite(avg) ? avg : null;
            const currentProb = Number.isFinite(mark) ? mark : null;
            const delta = Number.isFinite(avg) && Number.isFinite(mark) ? mark - avg : null;
            const entryMana = Number.isFinite(avg) ? avg * shares : null;
            const currentMana = Number.isFinite(mark) ? mark * shares : null;
            const winValue = Number.isFinite(shares) ? shares : null;
            const pnlValue = Number.isFinite(position.pnl)
              ? position.pnl
              : Number.isFinite(avg) && Number.isFinite(mark)
              ? (mark - avg) * shares
              : null;
            const deltaClass = delta == null ? "" : delta >= 0 ? "positive" : "negative";
            const pnlValueClass = pnlValue == null ? "" : pnlValue >= 0 ? "positive" : "negative";
            return `
        <li>
          <div class="position-market">
            <p class="position-title">${position.question}</p>
            <p class="position-meta">${position.outcome}</p>
          </div>
          <div class="position-cell">
            <span class="cell-label">Bought</span>
            <span class="cell-value">${formatMana(entryMana)} mana</span>
            <span class="cell-sub">${formatProb(entryProb)}</span>
          </div>
          <div class="position-cell">
            <span class="cell-label">Current</span>
            <span class="cell-value">${formatMana(currentMana)} mana</span>
            <span class="cell-sub">${formatProb(currentProb)}</span>
          </div>
          <div class="position-cell">
            <span class="cell-label">Win Value</span>
            <span class="cell-value">${formatMana(winValue)} mana</span>
            <span class="cell-sub">If ${position.outcome} resolves</span>
          </div>
          <div class="position-cell ${deltaClass}">
            <span class="cell-label">Δ Prob</span>
            <span class="cell-value">${formatDelta(delta)}</span>
          </div>
          <div class="position-cell ${pnlValueClass}">
            <span class="cell-label">PnL</span>
            <span class="cell-value">${formatSignedMana(pnlValue)}</span>
          </div>
        </li>`;
          }
        )
        .join("")}
    </ul>
  `;
}

function renderDetail(agent) {
  if (!agent) {
    detailTitle.textContent = "Select an agent";
    detailSubtitle.textContent = "Choose an agent from the leaderboard to see their positions and logs.";
    detailTitle.style.color = "";
    detailSubtitle.style.color = "";
    detailContainer.removeAttribute("style");
    detailContainer.innerHTML = `<p class="empty-state">No agent selected.</p>`;
    return;
  }
  detailTitle.textContent = agent.name;
  const latestRun = agent.history && agent.history.length ? agent.history[agent.history.length - 1].date : "n/a";
  detailSubtitle.textContent = `${agent.provider} • ${agent.model} • Wallet ${agent.wallet} • Most recent run ${latestRun}`;
  detailTitle.style.color = agent.color || "";
  detailSubtitle.style.color = agent.color || "";
  detailContainer.setAttribute(
    "style",
    `--detail-accent-strong: ${agent.color || "var(--accent)"}; --detail-accent-soft: ${
      agent.colorMuted || "var(--accent-muted)"
    };`
  );
  detailContainer.innerHTML = `
    <div class="detail-grid single">
      <div class="profile-stats profile-stats-inline">
        <div class="profile-stat">
          <span class="profile-stat-label">Bankroll</span>
          <span class="profile-stat-value">${currency(agent.bankroll)}</span>
        </div>
        <div class="profile-stat">
          <span class="profile-stat-label">Cash</span>
          <span class="profile-stat-value">${currency(agent.cash)}</span>
        </div>
        <div class="profile-stat">
          <span class="profile-stat-label">Total Assets</span>
          <span class="profile-stat-value">${currency(agent.totalAssets || agent.bankroll)}</span>
        </div>
      </div>
    </div>
    <article class="detail-card">
      <h3>Current Positions</h3>
      ${renderPositions(agent.positions)}
    </article>
    <article class="detail-card accent">
      <h3>Trade Log</h3>
      ${renderTradesTable(agent.trades)}
    </article>
  `;
}

function renderLoadingState() {
  return `
    <div class="loading-state">
      <span class="spinner" aria-hidden="true"></span>
      <span>Loading live data…</span>
    </div>
  `;
}

function selectAgent(slug) {
  activeSlug = slug;
  renderAgentGrid();
  const agent = agents.find((entry) => entry.slug === slug);
  renderDetail(agent);
}

const params = new URLSearchParams(window.location.search);
const apiOverride = params.get("api");
const apiEndpoint = apiOverride || "/api/live-runs";
const fallbackEndpoint = "data/live_runs.json";
const refreshMs = Number(params.get("refresh") || 30000);

async function loadPayload() {
  const endpoints = [apiEndpoint, fallbackEndpoint];
  let lastError = null;
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function bootstrap() {
  try {
    detailContainer.innerHTML = renderLoadingState();
    const payload = await loadPayload();
    agents = payload.agents || [];
    renderSummary(payload.summary || {}, payload.lastUpdated || new Date().toISOString());
    renderAgentGrid();
    if (agents.length) {
      selectAgent(agents[0].slug);
    }
  } catch (error) {
    detailContainer.innerHTML = `<p class="empty-state">Unable to load data: ${error}</p>`;
  }
}

bootstrap();
setInterval(bootstrap, refreshMs);
