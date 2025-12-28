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
  const { slug, name, provider, model, cash, dailyPnl, winRate, openPositions, color, colorMuted } = agent;
  const pnlSign = dailyPnl >= 0 ? "+" : "";
  const selected = slug === activeSlug ? "active" : "";
  const accentStyle = `style="--agent-accent: ${color || "var(--accent)"}; --agent-accent-muted: ${
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

function renderHistory(history = []) {
  if (!history.length) {
    return "<p class='empty-state'>No completed runs yet.</p>";
  }
  return `
    <ul>
      ${history
        .map(
          (entry) => `
        <li class="stat-row">
          <span>${entry.date}</span>
          <strong>${entry.pnl >= 0 ? "+" : ""}${currency(entry.pnl)}</strong>
        </li>`
        )
        .join("")}
    </ul>
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
  detailSubtitle.textContent = `${agent.provider} • ${agent.model} • Wallet ${agent.wallet}`;
  detailTitle.style.color = agent.color || "";
  detailSubtitle.style.color = agent.color || "";
  detailContainer.setAttribute(
    "style",
    `--detail-accent: ${agent.color || "var(--accent)"}; --detail-accent-muted: ${
      agent.colorMuted || "var(--accent-muted)"
    };`
  );
  detailContainer.innerHTML = `
    <div class="detail-grid">
      <article class="detail-card accent">
        <h3>Profile</h3>
        <p>${agent.notes}</p>
        <p><strong>Bankroll:</strong> ${currency(agent.bankroll)}</p>
        <p><strong>Cash:</strong> ${currency(agent.cash)}</p>
      </article>
      <article class="detail-card">
        <h3>Recent Runs</h3>
        ${renderHistory(agent.history)}
      </article>
    </div>
    <article class="detail-card accent">
      <h3>Trade Log</h3>
      ${renderTradesTable(agent.trades)}
    </article>
  `;
}

function selectAgent(slug) {
  activeSlug = slug;
  renderAgentGrid();
  const agent = agents.find((entry) => entry.slug === slug);
  renderDetail(agent);
}

async function bootstrap() {
  try {
    const response = await fetch("data/mock_runs.json");
    const payload = await response.json();
    agents = payload.agents;
    renderSummary(payload.summary, payload.lastUpdated);
    renderAgentGrid();
    if (agents.length) {
      selectAgent(agents[0].slug);
    }
  } catch (error) {
    detailContainer.innerHTML = `<p class="empty-state">Unable to load mock data: ${error}</p>`;
  }
}

bootstrap();
