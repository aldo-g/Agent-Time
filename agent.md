# Agent-Time Plan

## Purpose & Scope
- **Goal:** build an autonomous Polymarket prediction-market agent that consistently seeks profit while logging all reasoning and keeping risk bounded.
- **North Star:** daily automated runs that evaluate markets, execute trades with clear decision packets, and publish PnL dashboards.
- **Approach:** iterate from read-only data access toward full execution, research, and learning loops while keeping interfaces pluggable for future arenas.

## System Architecture Overview
1. **Core Agent Interface** – Environment-agnostic observation/action/reward contract, shared types, and decision-packet logging hooks.
2. **Connectors** – Polymarket client (read-only → execution) plus future BaseConnector for paper venues.
3. **Risk Engine** – Hard rule checks (bet sizing, exposure, stop-loss, liquidity filters, kill switch) enforced before trades.
4. **Database Layer** – Postgres migrations + ORM models for runs, market snapshots, decisions, evidence, trades, positions, pnl snapshots.
5. **Agents** – Baseline trading logic (momentum/mean reversion/etc.), later research agent for belief estimation and evidence gathering.
6. **Runner Automation** – Daily job orchestrating market fetch → evaluation → execution → persistence → portfolio refresh.
7. **Web Dashboard** – Next.js app surfacing runs, decision feed/detail, performance charts, calibration stats.
8. **Learning / Adaptation** – Reflection module updating strategy state from realized PnL and signal performance.

## Milestones & Deliverables
| Step | Description | Key Deliverable |
| --- | --- | --- |
| 0 | Core agent interface/types | `core/agent_interface.py` |
| 1 | Polymarket read-only connector | `connectors/polymarket.py` (list/get markets/fills) |
| 2 | Execution + portfolio APIs | Connector methods: `placeBet`, `sellPosition`, `getPortfolio*`, `getTransactions` |
| 3 | Database + decision packet schema | SQL migrations + ORM models |
| 4 | Baseline profit agent | `agents/baseline_agent.py` |
| 5 | Risk engine | `risk/rules.py` |
| 6 | Daily runner automation | `runner/daily.py` + scheduler config |
| 7 | Web dashboard MVP | `/web` Next.js pages for runs/decisions/performance |
| 8 | Research agent + evidence collector | `agents/research_agent.py`, `evidence_collector.py` |
| 9 | Learning/reflection loop | `learning/reflection.py` + `strategy_state` table |
| 10 | Generic connector interface | `connectors/base_connector.py` |

## Interfaces & Schemas (initial notes)
- **Agent Observation:** market metadata, current prob, liquidity/activity stats, timestamp, existing positions/cash.
- **Agent Action:** bet (marketId, outcome, size), sell (marketId, shares or amount), noop.
- **Rewards:** realized/unrealized PnL deltas captured via portfolio snapshots.
- **Decision Packet Fields:** market info, observed prob, agent belief + confidence, estimated EV/edge, bet sizing inputs, constraint outcomes, action, rationale text, evidence references.
- **Connector API Surface:** methods listed above, all returning typed responses + normalized error handling.

## Open Questions / Inputs Needed
- Polymarket wallet/API credentials for authenticated endpoints (Steps 2+) — fill in `POLYMARKET_WALLET_ADDRESS`, `POLYMARKET_API_KEY`, `POLYMARKET_SECRET`, `POLYMARKET_PASSPHRASE`.
- Postgres deployment target + connection string (local vs hosted).
- Preferred ORM (Prisma, TypeORM, Drizzle, etc.) + migration tooling.
- Hosting for runner + dashboard (e.g., Fly.io, Railway, Vercel, self-hosted?).
- Monitoring/alerting requirements (kill switch, failure notifications).

## Next Actions
1. Implement Step 0: scaffold `core/agent_interface.py` with types and decision-packet structure.
2. Stand up read-only Polymarket connector (Step 1) to unblock downstream layers.
3. Decide on DB/ORM stack to begin Step 3 shortly after connector types stabilize.
