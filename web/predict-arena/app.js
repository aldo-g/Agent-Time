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

function parseDateString(dateText) {
  const [year, month, day] = String(dateText || "")
    .split("-")
    .map((part) => Number(part));
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) {
    return null;
  }
  return new Date(year, month - 1, day);
}

function formatDateShort(dateText) {
  const parsed = parseDateString(dateText);
  if (!parsed) return String(dateText || "n/a");
  return parsed.toLocaleDateString([], { month: "short", day: "numeric" });
}

function buildValueSeries(agent) {
  const history = Array.isArray(agent.history) ? agent.history : [];
  const entries = history
    .map((entry) => {
      const bankroll = Number(entry.bankroll);
      const pnl = Number(entry.pnl);
      return {
        date: String(entry.date || ""),
        bankroll: Number.isFinite(bankroll) ? bankroll : null,
        pnl: Number.isFinite(pnl) ? pnl : 0,
      };
    })
    .filter((entry) => entry.date)
    .sort((a, b) => a.date.localeCompare(b.date));

  if (!entries.length) return [];
  const bankrollSeries = entries.filter((entry) => Number.isFinite(entry.bankroll));
  if (bankrollSeries.length === entries.length) {
    return bankrollSeries.map((entry) => ({ date: entry.date, value: entry.bankroll }));
  }

  const current = Number(agent.totalAssets ?? agent.bankroll);
  if (!Number.isFinite(current)) return [];
  const totalChange = entries.reduce((sum, entry) => sum + entry.pnl, 0);
  const startingValue = current - totalChange;
  let cumulative = 0;
  return entries.map((entry) => {
    cumulative += entry.pnl;
    return { date: entry.date, value: startingValue + cumulative };
  });
}

function renderValueChart(agent) {
  const series = buildValueSeries(agent);
  if (series.length < 2) {
    return `
      <section class="detail-card value-card">
        <div class="value-card-header">
          <h3>Value Over Time</h3>
        </div>
        <p class="empty-state">Need at least 2 daily snapshots to draw the value trend.</p>
      </section>
    `;
  }

  const chartWidth = 860;
  const chartHeight = 260;
  const padding = { top: 16, right: 16, bottom: 32, left: 56 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  const values = series.map((entry) => entry.value);
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (minValue === maxValue) {
    const buffer = Math.max(1, Math.abs(minValue) * 0.02);
    minValue -= buffer;
    maxValue += buffer;
  }

  const toX = (index) =>
    padding.left + (series.length === 1 ? 0 : (index / (series.length - 1)) * innerWidth);
  const toY = (value) =>
    padding.top + ((maxValue - value) / (maxValue - minValue)) * innerHeight;

  const points = series.map((entry, index) => ({
    ...entry,
    x: toX(index),
    y: toY(entry.value),
  }));
  const baselineY = padding.top + innerHeight;
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} ${baselineY.toFixed(2)} L ${points[0].x.toFixed(2)} ${baselineY.toFixed(2)} Z`;

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const ratio = index / (tickCount - 1);
    const value = maxValue - ratio * (maxValue - minValue);
    return {
      value,
      y: toY(value),
    };
  });

  const first = series[0];
  const last = series[series.length - 1];
  const change = last.value - first.value;
  const changePercent = first.value > 0 ? change / first.value : null;
  const changeClass = change > 0 ? "positive" : change < 0 ? "negative" : "";

  return `
    <section class="detail-card value-card">
      <div class="value-card-header">
        <h3>Value Over Time</h3>
        <div class="value-card-stats">
          <span>Start ${currency(Math.round(first.value))}</span>
          <span>Current ${currency(Math.round(last.value))}</span>
          <span class="${changeClass}">Change ${formatSignedMana(change)} (${formatPercent(changePercent)})</span>
        </div>
      </div>
      <div class="value-chart-shell">
        <svg class="value-chart" viewBox="0 0 ${chartWidth} ${chartHeight}" role="img" aria-label="Total assets over time chart">
          ${ticks
            .map(
              (tick) => `
            <line x1="${padding.left}" y1="${tick.y.toFixed(2)}" x2="${(padding.left + innerWidth).toFixed(2)}" y2="${tick.y.toFixed(2)}" class="value-grid-line" />
            <text x="${(padding.left - 8).toFixed(2)}" y="${(tick.y + 4).toFixed(2)}" text-anchor="end" class="value-axis-text">${formatMana(tick.value)}</text>
          `
            )
            .join("")}
          <path d="${areaPath}" class="value-area" />
          <path d="${linePath}" class="value-line" />
          ${points
            .map(
              (point) => `
            <circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="3.2" class="value-point">
              <title>${formatDateShort(point.date)}: ${formatMana(point.value)} mana</title>
            </circle>
          `
            )
            .join("")}
        </svg>
        <div class="value-axis-dates">
          <span>${formatDateShort(first.date)}</span>
          <span>${formatDateShort(last.date)}</span>
        </div>
      </div>
    </section>
  `;
}

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
  const status = String(trade.status || "").toUpperCase();
  if (status === "OPEN") {
    return `<span class="pill open">Open</span>`;
  }
  if (status === "EXECUTED") {
    return `<span class="pill executed">Executed</span>`;
  }
  if (status === "SKIPPED") {
    return `<span class="pill skipped">Skipped</span>`;
  }
  if (status === "FAILED") {
    return `<span class="pill failed">Failed</span>`;
  }
  return `<span class="pill">${status || "Unknown"}</span>`;
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
              const marketLabel = trade.marketUrl
                ? `<a href="${trade.marketUrl}" target="_blank" rel="noreferrer">${trade.market}</a>`
                : `<span>${trade.market}</span>`;
              const tradeTime = trade.timestamp
                ? new Date(trade.timestamp).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
                : "n/a";
              return `
                <tr>
                  <td>${tradeTime}</td>
                  <td>
                    ${marketLabel}
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
    ${renderValueChart(agent)}
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

async function loadPayload() {
  const response = await fetch(apiEndpoint, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return await response.json();
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
    const walletNote = agent.wallet ? ` • Wallet ${agent.wallet}` : "";
    agentMetaEl.textContent = `${agent.provider} • ${agent.model}${walletNote} • Last run ${latestRun}`;

    renderHeaderStats(agent);
    renderDetail(agent);

    if (payload.lastUpdated) {
      footerUpdated.textContent = `Last updated: ${new Date(payload.lastUpdated).toLocaleString()}`;
    }
  } catch (error) {
    agentNameEl.textContent = "Error";
    detailContainer.innerHTML = `<p class="empty-state">Unable to load API data from ${apiEndpoint}: ${error}</p>`;
  }
}

bootstrap();
