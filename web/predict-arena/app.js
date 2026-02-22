const agentNameEl = document.getElementById("agent-name");
const agentMetaEl = document.getElementById("agent-meta");
const headerStatsEl = document.getElementById("header-stats");
const detailContainer = document.getElementById("agent-details");
const footerUpdated = document.getElementById("footer-updated");

const currency = (value) => {
  const formatted = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
  return `<img class="mana-icon" src="assets/Mana-Logo.svg" alt="Mana" />${formatted}`;
};

const formatMana = (value) =>
  value == null ? "n/a" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
const formatSignedMana = (value) => {
  if (value == null) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${formatMana(Math.abs(value))}`;
};
const formatProb = (value) => (value == null ? "n/a" : `${(value * 100).toFixed(1)}%`);
const formatDelta = (value) => {
  if (value == null) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${(Math.abs(value) * 100).toFixed(1)}pp`;
};
const formatPercent = (value) => {
  if (value == null || Number.isNaN(value)) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${(Math.abs(value) * 100).toFixed(1)}%`;
};

function totalGainPercent(agent) {
  const history = Array.isArray(agent.history) ? agent.history : [];
  const totalChange = history.reduce((sum, entry) => {
    const pnl = Number(entry.pnl);
    return Number.isFinite(pnl) ? sum + pnl : sum;
  }, 0);
  const bankroll = Number(agent.bankroll);
  if (!Number.isFinite(bankroll)) return null;
  const initial = bankroll - totalChange;
  if (!Number.isFinite(initial) || initial <= 0) return null;
  return totalChange / initial;
}

function renderHeaderStats(agent) {
  const gainPct = totalGainPercent(agent);
  const gainClass = gainPct == null ? "" : gainPct >= 0 ? "positive" : "negative";
  const cash = Math.round(Number(agent.cash || 0));
  const totalAssets = Math.round(Number(agent.totalAssets || agent.bankroll || 0));
  const positionsValue = agent.positionsValue != null
    ? Math.round(Number(agent.positionsValue))
    : totalAssets - cash;

  headerStatsEl.innerHTML = `
    <div class="header-stat">
      <span class="header-stat-label">Total Assets</span>
      <span class="header-stat-value">${currency(totalAssets)}</span>
    </div>
    <div class="header-stat">
      <span class="header-stat-label">Cash</span>
      <span class="header-stat-value">${currency(cash)}</span>
    </div>
    <div class="header-stat">
      <span class="header-stat-label">Invested</span>
      <span class="header-stat-value">${currency(positionsValue)}</span>
    </div>
    <div class="header-stat">
      <span class="header-stat-label">Total Gain</span>
      <span class="header-stat-value ${gainClass}">${formatPercent(gainPct)}</span>
    </div>
    <div class="header-stat">
      <span class="header-stat-label">Open Positions</span>
      <span class="header-stat-value">${agent.openPositions ?? 0}</span>
    </div>
  `;
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
            <th>Market / Rationale</th>
            <th>Action</th>
            <th>Stake</th>
            <th>Prob Δ</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${trades
            .map((trade) => {
              const probAfter = Number(trade.probAfter);
              const probBefore = Number(trade.probBefore);
              const delta = Number.isFinite(probAfter) && Number.isFinite(probBefore) ? probAfter - probBefore : 0;
              const tools = Array.isArray(trade.tools) ? trade.tools.filter(Boolean) : [];
              const sources = Array.isArray(trade.sources) ? trade.sources.filter(Boolean) : [];
              const toolList = tools.length
                ? `<div class="tool-list">${tools.map((tool) => `<span class="pill tool-pill">${tool}</span>`).join("")}</div>`
                : "";
              const sourceList = sources.length
                ? `<div class="source-list">${sources
                    .map((url) => {
                      const label = url.replace(/^https?:\/\//i, "").replace(/\/$/, "");
                      return `<a class="pill source-pill" href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
                    })
                    .join("")}</div>`
                : "";
              const reason = trade.reason ? String(trade.reason) : "";
              const reasonBlock = reason ? `<p class="trade-reason">${reason}</p>` : "";
              return `
                <tr>
                  <td>${new Date(trade.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
                  <td>
                    <a href="${trade.marketUrl}" target="_blank" rel="noreferrer">${trade.market}</a>
                    ${reasonBlock}
                    ${sourceList}
                    ${toolList}
                  </td>
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
        .map((position) => {
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
        })
        .join("")}
    </ul>
  `;
}

function renderDetail(agent) {
  detailContainer.innerHTML = `
    <section class="detail-card tabbed-card">
      <div class="tab-header">
        <h3>Positions & Trades</h3>
        <div class="tab-buttons" role="tablist" aria-label="Agent detail tabs">
          <button class="tab-button active" role="tab" aria-selected="true" data-tab="positions">Current Positions</button>
          <button class="tab-button" role="tab" aria-selected="false" data-tab="trades">Trade Log</button>
        </div>
      </div>
      <div class="tab-panels">
        <div class="tab-panel active" role="tabpanel" data-panel="positions">
          ${renderPositions(agent.positions)}
        </div>
        <div class="tab-panel" role="tabpanel" data-panel="trades" hidden>
          ${renderTradesTable(agent.trades)}
        </div>
      </div>
    </section>
  `;
  setupTabs(detailContainer);
}

function setupTabs(container) {
  const tabButtons = container.querySelectorAll(".tab-button");
  const tabPanels = container.querySelectorAll(".tab-panel");
  if (!tabButtons.length || !tabPanels.length) return;
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tab;
      tabButtons.forEach((btn) => {
        const isActive = btn === button;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", String(isActive));
      });
      tabPanels.forEach((panel) => {
        const isActive = panel.dataset.panel === target;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
      });
    });
  });
}

const params = new URLSearchParams(window.location.search);
const apiOverride = params.get("api");
const apiEndpoint = apiOverride || "/api/live-runs?refresh=1";
const fallbackEndpoint = "data/live_runs.json";

async function loadPayload() {
  const endpoints = [apiEndpoint, fallbackEndpoint];
  let lastError = null;
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function bootstrap() {
  try {
    const payload = await loadPayload();
    const agent = (payload.agents || [])[0];
    if (!agent) {
      detailContainer.innerHTML = `<p class="empty-state">No agent data found.</p>`;
      return;
    }

    agentNameEl.textContent = agent.name;
    const latestRun = agent.history && agent.history.length
      ? agent.history[agent.history.length - 1].date
      : "n/a";
    agentMetaEl.textContent = `${agent.provider} • ${agent.model} • Wallet ${agent.wallet} • Last run ${latestRun}`;

    renderHeaderStats(agent);
    renderDetail(agent);

    if (payload.lastUpdated) {
      footerUpdated.textContent = `Last updated: ${new Date(payload.lastUpdated).toLocaleString()}`;
    }
  } catch (error) {
    agentNameEl.textContent = "Error";
    detailContainer.innerHTML = `<p class="empty-state">Unable to load data: ${error}</p>`;
  }
}

bootstrap();
