const TOOL_DESCRIPTIONS = {
  manifold_portfolio: "Fetches the agent's current Manifold portfolio — cash balance, open positions, and unrealised P&L",
  portfolio_analytics: "Analyses position sizing, concentration, and overall portfolio risk metrics",
  manifold_markets: "Searches Manifold for prediction markets matching a query or topic",
  manifold_market_details: "Retrieves full details for a specific market: description, resolution criteria, current probability",
  manifold_market_history: "Fetches historical probability data and trading volume for a market",
  manifold_place_bet: "Places a real bet on Manifold — buys YES or NO shares in a market",
  manifold_sell_position: "Sells an existing position to realise gains or cut losses",
  limit_order_preview: "Previews the expected shares and cost of a limit order before placing it",
  risk_gate: "Kelly-criterion risk gate — checks whether a trade size is within safe bankroll limits",
  duckduckgo_search: "Searches the web via DuckDuckGo to gather news and context for a market",
  web_scrape: "Scrapes a specific webpage to extract text content for analysis",
  event_timer: "Checks the current date/time and calculates time remaining until a market's resolution deadline",
};

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
  const gradientId = "areaGrad";

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
        <svg class="value-chart" id="value-chart-svg" viewBox="0 0 ${chartWidth} ${chartHeight}" role="img" aria-label="Total assets over time chart">
          <defs>
            <linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="rgb(159,75,58)" stop-opacity="0.22"/>
              <stop offset="100%" stop-color="rgb(159,75,58)" stop-opacity="0"/>
            </linearGradient>
          </defs>
          ${ticks
            .map(
              (tick) => `
            <line x1="${padding.left}" y1="${tick.y.toFixed(2)}" x2="${(padding.left + innerWidth).toFixed(2)}" y2="${tick.y.toFixed(2)}" class="value-grid-line" />
            <text x="${(padding.left - 8).toFixed(2)}" y="${(tick.y + 4).toFixed(2)}" text-anchor="end" class="value-axis-text">${formatMana(tick.value)}</text>
          `
            )
            .join("")}
          <path d="${areaPath}" class="value-area" fill="url(#${gradientId})" />
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
          <g class="chart-tooltip-group" style="display:none">
            <line class="chart-tooltip-line" x1="0" y1="${padding.top}" x2="0" y2="${baselineY}"/>
            <circle class="chart-tooltip-dot" r="5"/>
            <rect class="chart-tooltip-box" rx="5" ry="5" width="110" height="36"/>
            <text class="chart-tooltip-text chart-tooltip-date" x="0" y="0" text-anchor="middle"/>
            <text class="chart-tooltip-text chart-tooltip-value" x="0" y="0" text-anchor="middle" font-weight="600"/>
          </g>
          <rect id="chart-hover-zone" x="${padding.left}" y="${padding.top}" width="${innerWidth}" height="${innerHeight}" fill="transparent" style="cursor:crosshair"/>
        </svg>
        <div class="value-axis-dates">
          <span>${formatDateShort(first.date)}</span>
          <span>${formatDateShort(last.date)}</span>
        </div>
      </div>
    </section>
  `;
}

function setupChartTooltip(points, padding, chartWidth) {
  const svg = document.getElementById("value-chart-svg");
  if (!svg) return;
  const hoverZone = svg.getElementById ? svg.querySelector("#chart-hover-zone") : null;
  const tooltipGroup = svg.querySelector(".chart-tooltip-group");
  if (!hoverZone || !tooltipGroup || !points.length) return;

  const tooltipLine = tooltipGroup.querySelector(".chart-tooltip-line");
  const tooltipDot = tooltipGroup.querySelector(".chart-tooltip-dot");
  const tooltipBox = tooltipGroup.querySelector(".chart-tooltip-box");
  const tooltipDate = tooltipGroup.querySelector(".chart-tooltip-date");
  const tooltipValue = tooltipGroup.querySelector(".chart-tooltip-value");

  const boxW = 110;
  const boxH = 36;

  hoverZone.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const scaleX = 860 / rect.width;
    const svgX = (e.clientX - rect.left) * scaleX;

    // find nearest point by X
    let closest = points[0];
    let minDist = Math.abs(points[0].x - svgX);
    for (const pt of points) {
      const d = Math.abs(pt.x - svgX);
      if (d < minDist) { minDist = d; closest = pt; }
    }

    tooltipLine.setAttribute("x1", closest.x.toFixed(2));
    tooltipLine.setAttribute("x2", closest.x.toFixed(2));
    tooltipDot.setAttribute("cx", closest.x.toFixed(2));
    tooltipDot.setAttribute("cy", closest.y.toFixed(2));

    // clamp box so it stays within viewBox
    const boxX = Math.min(Math.max(closest.x - boxW / 2, padding.left), chartWidth - padding.right - boxW);
    const boxY = Math.max(closest.y - boxH - 10, padding.top);
    tooltipBox.setAttribute("x", boxX.toFixed(2));
    tooltipBox.setAttribute("y", boxY.toFixed(2));

    const textX = (boxX + boxW / 2).toFixed(2);
    tooltipDate.setAttribute("x", textX);
    tooltipDate.setAttribute("y", (boxY + 13).toFixed(2));
    tooltipDate.textContent = formatDateShort(closest.date);

    tooltipValue.setAttribute("x", textX);
    tooltipValue.setAttribute("y", (boxY + 27).toFixed(2));
    tooltipValue.textContent = `${formatMana(closest.value)} mana`;

    tooltipGroup.style.display = "";
  });

  hoverZone.addEventListener("mouseleave", () => {
    tooltipGroup.style.display = "none";
  });
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
    <div class="trades-list">
      ${trades
        .map((trade) => {
          const probAfter = Number(trade.probAfter);
          const probBefore = Number(trade.probBefore);
          const delta = Number.isFinite(probAfter) && Number.isFinite(probBefore) ? probAfter - probBefore : null;
          const tools = Array.isArray(trade.tools) ? trade.tools.filter(Boolean) : [];
          const sources = Array.isArray(trade.sources) ? trade.sources.filter(Boolean) : [];
          const sourcePills = sources.map((url) => {
            const label = url.replace(/^https?:\/\//i, "").replace(/\/$/, "");
            return `<a class="pill source-pill" href="${url}" target="_blank" rel="noreferrer">${label}</a>`;
          });
          const toolPills = tools.map((tool) => {
            const desc = TOOL_DESCRIPTIONS[tool];
            return `<span class="pill tool-pill" ${desc ? `title="${desc}"` : ""}>${tool}</span>`;
          });
          const reason = trade.reason ? String(trade.reason) : "";
          const marketTitle = trade.marketUrl
            ? `<a href="${trade.marketUrl}" target="_blank" rel="noreferrer">${trade.market}</a>`
            : trade.market;
          const tradeTime = trade.timestamp
            ? new Date(trade.timestamp).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
            : "n/a";
          const deltaStr = delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp` : "—";
          return `
            <div class="trade-card">
              <p class="trade-card-title">${marketTitle}</p>
              ${reason ? `<p class="trade-card-reason">${reason}</p>` : ""}
              <div class="trade-card-meta">
                <span>${tradeTime}</span>
                <span><strong>${trade.action} ${trade.outcome}</strong></span>
                <span>${currency(trade.amount)}</span>
                <span>Δ ${deltaStr}</span>
                ${tradeStatusPill(trade)}
              </div>
              ${sourcePills.length ? `<div class="trade-card-pills trade-card-sources"><span class="trade-card-pill-label">Sources</span>${sourcePills.join("")}</div>` : ""}
              ${toolPills.length ? `<div class="trade-card-pills trade-card-tools"><span class="trade-card-pill-label">Tools</span>${toolPills.join("")}</div>` : ""}
            </div>
          `;
        })
        .join("")}
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
      <span>Probability</span>
      <span>Bought</span>
      <span>Win Value</span>
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
          const winValue = Number.isFinite(shares) ? shares : null;
          const pnlValue = Number.isFinite(position.pnl)
            ? position.pnl
            : Number.isFinite(avg) && Number.isFinite(mark)
            ? (mark - avg) * shares
            : null;
          const rowClass = pnlValue == null ? "" : pnlValue >= 0 ? "row-positive" : "row-negative";

          // Probability bar
          const entryPct = entryProb != null ? entryProb * 100 : null;
          const currentPct = currentProb != null ? currentProb * 100 : null;
          const moved = delta != null ? (delta > 0.001 ? "up" : delta < -0.001 ? "down" : "flat") : "flat";
          const barFillColor = moved === "up" ? "var(--positive)" : moved === "down" ? "var(--negative)" : "var(--muted)";
          const arrowSymbol = moved === "up" ? "▲" : moved === "down" ? "▼" : "";
          const arrowClass = moved === "up" ? "positive" : moved === "down" ? "negative" : "";
          const deltaLabel = delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp` : "";

          const probBar = entryPct != null && currentPct != null ? `
            <div class="prob-bar-combined" title="Entry: ${entryPct.toFixed(1)}% → Current: ${currentPct.toFixed(1)}%">
              <div class="prob-bar-track-v2">
                <div class="prob-bar-fill-v2" style="width:${currentPct.toFixed(1)}%; background:${barFillColor};"></div>
                <div class="prob-bar-marker" style="left:${entryPct.toFixed(1)}%;"></div>
              </div>
              <div class="prob-bar-labels">
                <span class="prob-bar-entry">
                  <span class="prob-bar-sublabel">Entry</span>
                  ${entryPct.toFixed(1)}%
                </span>
                ${arrowSymbol ? `<span class="prob-bar-arrow ${arrowClass}">${arrowSymbol} ${deltaLabel}</span>` : ""}
                <span class="prob-bar-current">
                  <span class="prob-bar-sublabel">Now</span>
                  ${currentPct.toFixed(1)}%
                </span>
              </div>
            </div>` : `<span class="cell-sub">n/a</span>`;

          return `
        <li class="${rowClass}">
          <div class="position-market">
            <p class="position-title">${position.question}</p>
            <p class="position-meta">${position.outcome}</p>
          </div>
          <div class="position-cell prob-bar-cell">
            ${probBar}
          </div>
          <div class="position-cell">
            <span class="cell-label">Bought</span>
            <span class="cell-value">${formatMana(entryMana)} mana</span>
          </div>
          <div class="position-cell">
            <span class="cell-label">Win Value</span>
            <span class="cell-value">${formatMana(winValue)} mana</span>
            <span class="cell-sub">If ${position.outcome} resolves</span>
          </div>
        </li>`;
        })
        .join("")}
    </ul>
  `;
}

function buildStatsStrip(agent) {
  const trades = Array.isArray(agent.trades) ? agent.trades : [];
  const resolved = trades.filter((t) => {
    const s = String(t.status || "").toUpperCase();
    return s === "EXECUTED" || s === "RESOLVED" || s === "LOSS";
  });
  const wins = resolved.filter((t) => {
    const s = String(t.status || "").toUpperCase();
    if (s === "LOSS") return false;
    const pnl = Number(t.pnl);
    return Number.isFinite(pnl) ? pnl > 0 : s !== "LOSS";
  }).length;
  const losses = resolved.filter((t) => {
    const s = String(t.status || "").toUpperCase();
    if (s === "LOSS") return true;
    const pnl = Number(t.pnl);
    return Number.isFinite(pnl) && pnl < 0;
  }).length;
  const open = trades.filter((t) => String(t.status || "").toUpperCase() === "OPEN").length;
  const winRate = resolved.length > 0 ? wins / resolved.length : null;
  const winRateClass = winRate == null ? "" : winRate >= 0.5 ? "positive" : "negative";

  return `
    <div class="stats-strip">
      <div class="stats-chip">
        <span class="stats-chip-label">Wins</span>
        <span class="stats-chip-value positive">${wins}</span>
      </div>
      <div class="stats-chip">
        <span class="stats-chip-label">Losses</span>
        <span class="stats-chip-value negative">${losses}</span>
      </div>
      <div class="stats-chip">
        <span class="stats-chip-label">Open</span>
        <span class="stats-chip-value">${open}</span>
      </div>
      <div class="stats-chip">
        <span class="stats-chip-label">Win Rate</span>
        <span class="stats-chip-value ${winRateClass}">${winRate != null ? `${(winRate * 100).toFixed(0)}%` : "—"}</span>
      </div>
    </div>
  `;
}

function renderDetail(agent) {
  const series = buildValueSeries(agent);
  const chartWidth = 860;
  const padding = { top: 16, right: 16, bottom: 32, left: 56 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = 260 - padding.top - padding.bottom;
  const values = series.map((e) => e.value);
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (minValue === maxValue) { const buf = Math.max(1, Math.abs(minValue) * 0.02); minValue -= buf; maxValue += buf; }
  const toX = (i) => padding.left + (series.length === 1 ? 0 : (i / (series.length - 1)) * innerWidth);
  const toY = (v) => padding.top + ((maxValue - v) / (maxValue - minValue)) * innerHeight;
  const points = series.map((e, i) => ({ ...e, x: toX(i), y: toY(e.value) }));

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
  setupChartTooltip(points, padding, chartWidth);
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
const apiEndpoint = apiOverride || "/api/live-runs?refresh=1&live=1";

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
    agentMetaEl.innerHTML = `${agent.provider} • ${agent.model}${walletNote}<span class="last-run-badge">Last run ${latestRun}</span>`;

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
