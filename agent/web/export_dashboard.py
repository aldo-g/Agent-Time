#!/usr/bin/env python3
"""Build the Predict Arena frontend payload from run logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Claude",
    "claude": "Claude",
    "gemini": "Gemini",
    "google": "Gemini",
}

PROVIDER_COLORS = {
    "openai": ("#3f7f5f", "rgba(63, 127, 95, 0.16)"),
    "anthropic": ("#b07a2a", "rgba(176, 122, 42, 0.18)"),
    "claude": ("#b07a2a", "rgba(176, 122, 42, 0.18)"),
    "gemini": ("#5b79a6", "rgba(91, 121, 166, 0.18)"),
    "google": ("#5b79a6", "rgba(91, 121, 166, 0.18)"),
}


@dataclass
class AgentConfig:
    name: str
    model_provider: str
    model: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentConfig":
        return cls(
            name=str(payload.get("name") or ""),
            model_provider=str(payload.get("model_provider") or ""),
            model=str(payload.get("model") or ""),
        )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_iso(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    normalized = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_trades_from_output(output: Any) -> List[Dict[str, str]]:
    """Extract trade + reason lines from agent output prose."""
    if not isinstance(output, str):
        return []
    trades: List[Dict[str, str]] = []
    last_trade: Dict[str, str] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        clean_line = line.strip("*_- ")
        match_trade = re.search(r"trade\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_trade:
            last_trade = {"trade": match_trade.group(1).strip("* "), "reason": ""}
            trades.append(last_trade)
            continue
        match_reason = re.search(r"reason\s*-\s*(.+)", clean_line, flags=re.IGNORECASE)
        if match_reason and last_trade is not None:
            last_trade["reason"] = match_reason.group(1).strip()
    return [
        {"trade": entry["trade"], "reason": entry.get("reason", "")}
        for entry in trades
        if entry.get("trade")
    ]


def _build_run_contexts(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for record in runs:
        dt = _parse_iso(record.get("timestamp"))
        if dt is None:
            continue
        rationales = _parse_trades_from_output(record.get("output"))
        tools = [str(entry) for entry in (record.get("tool_calls") or []) if entry]
        contexts.append({"timestamp": dt, "rationales": rationales, "tools": tools})
    contexts.sort(key=lambda item: item["timestamp"])
    return contexts


def _match_run_context(
    trade_dt: Optional[datetime],
    contexts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not contexts:
        return None
    if trade_dt is not None:
        candidates = [ctx for ctx in contexts if ctx["timestamp"] <= trade_dt]
        if candidates:
            return candidates[-1]
    return contexts[-1]


def _find_trade_reason(trade: Dict[str, Any], rationales: List[Dict[str, str]]) -> str:
    if not rationales:
        return ""
    question = str(trade.get("market") or "").lower()
    outcome = str(trade.get("outcome") or "").lower()
    market_id = str(trade.get("market_id") or "").lower()
    for entry in rationales:
        text = f"{entry.get('trade', '')} {entry.get('reason', '')}".lower()
        if question and question in text:
            return entry.get("reason") or entry.get("trade") or ""
        if market_id and market_id in text:
            return entry.get("reason") or entry.get("trade") or ""
        if outcome and outcome in text and question and question[:24] in text:
            return entry.get("reason") or entry.get("trade") or ""
    return rationales[0].get("reason") or rationales[0].get("trade") or ""


def _extract_links(text: str) -> List[str]:
    if not text:
        return []
    links: List[str] = []
    for match in re.finditer(r"https?://[^\s)]+", text):
        url = match.group(0).rstrip(".,);]")
        links.append(url)
    seen = set()
    deduped = []
    for url in links:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    slug = "-".join(part for part in cleaned.split("-") if part)
    return slug or "agent"


def _count_markets(payload: Any) -> int:
    if not payload:
        return 0
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return 0
    total = 0
    for event in events:
        if isinstance(event, dict):
            markets = event.get("markets")
            if isinstance(markets, list):
                total += len(markets)
                continue
        total += 1
    return total


def _group_latest_snapshots(runs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for record in runs:
        agent = record.get("agent")
        snapshot = record.get("portfolio")
        if not agent or not isinstance(snapshot, dict):
            continue
        timestamp = record.get("timestamp")
        dt = _parse_iso(timestamp)
        if dt is None:
            continue
        existing = latest.get(agent)
        if not existing or dt > existing["timestamp"]:
            latest[agent] = {"timestamp": dt, "snapshot": snapshot}
    return latest


def _build_history(
    runs: Iterable[Dict[str, Any]],
    trades_by_date: Dict[str, int],
) -> List[Dict[str, Any]]:
    latest_by_date: Dict[str, Dict[str, Any]] = {}
    for record in runs:
        snapshot = record.get("portfolio")
        if not isinstance(snapshot, dict):
            continue
        dt = _parse_iso(record.get("timestamp"))
        if dt is None:
            continue
        date_key = dt.date().isoformat()
        bankroll = snapshot.get("bankroll")
        if bankroll is None:
            continue
        existing = latest_by_date.get(date_key)
        if not existing or dt > existing["timestamp"]:
            latest_by_date[date_key] = {"timestamp": dt, "bankroll": float(bankroll)}
    history: List[Dict[str, Any]] = []
    prev_bankroll: Optional[float] = None
    for date_key in sorted(latest_by_date.keys()):
        bankroll = latest_by_date[date_key]["bankroll"]
        pnl = bankroll - prev_bankroll if prev_bankroll is not None else 0.0
        history.append(
            {
                "date": date_key,
                "pnl": pnl,
                "trades": trades_by_date.get(date_key, 0),
            }
        )
        prev_bankroll = bankroll
    return history


def _build_trades(
    trades: Iterable[Dict[str, Any]],
    agent: str,
    runs: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rendered: List[Dict[str, Any]] = []
    contexts = _build_run_contexts(runs)
    for entry in trades:
        timestamp = entry.get("timestamp")
        market_id = entry.get("market_id") or ""
        bet_id = entry.get("bet_id")
        trade_id = bet_id or f"{agent}-{timestamp}-{market_id}"
        trade_dt = _parse_iso(timestamp)
        context = _match_run_context(trade_dt, contexts)
        tools = context["tools"] if context else []
        reason = _find_trade_reason(entry, context["rationales"]) if context else ""
        sources = _extract_links(reason)
        rendered.append(
            {
                "id": trade_id,
                "timestamp": timestamp,
                "market": entry.get("market") or market_id,
                "marketUrl": entry.get("market_url") or "",
                "action": entry.get("action") or "BUY",
                "outcome": entry.get("outcome") or "",
                "amount": float(entry.get("amount") or 0.0),
                "probBefore": entry.get("prob_before"),
                "probAfter": entry.get("prob_after"),
                "status": entry.get("status") or "OPEN",
                "reason": reason,
                "tools": tools,
                "sources": sources,
            }
        )
    rendered.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return rendered


def _sum_mana_in_play(trades: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for entry in trades:
        status = str(entry.get("status") or "").upper()
        if status != "OPEN":
            continue
        amount = entry.get("amount")
        try:
            total += abs(float(amount))
        except (TypeError, ValueError):
            continue
    return total


def _latest_timestamp(*collections: Iterable[Dict[str, Any]]) -> str:
    latest: Optional[datetime] = None
    for collection in collections:
        for entry in collection:
            dt = _parse_iso(entry.get("timestamp"))
            if dt is None:
                continue
            if latest is None or dt > latest:
                latest = dt
    if latest is None:
        return datetime.now(timezone.utc).isoformat()
    return latest.isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="agents.json", help="Path to agents.json.")
    parser.add_argument("--results", default="results/multi_agent_runs.jsonl", help="JSONL run log path.")
    parser.add_argument("--trades", default="results/trades.jsonl", help="JSONL trade log path.")
    parser.add_argument("--markets", default="data/shared_markets.json", help="Shared market cache path.")
    parser.add_argument(
        "--out",
        default="web/predict-arena/data/live_runs.json",
        help="Output payload path for the frontend.",
    )
    return parser.parse_args()


def build_payload(
    *,
    agents_path: Path,
    results_path: Path,
    trades_path: Path,
    markets_path: Path,
) -> Dict[str, Any]:
    agents_payload = _load_json(agents_path) or []
    agent_configs = [
        AgentConfig.from_dict(entry)
        for entry in agents_payload
        if isinstance(entry, dict) and entry.get("name")
    ]

    runs = _load_jsonl(results_path)
    trades = _load_jsonl(trades_path)

    runs_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in runs:
        agent = record.get("agent")
        if agent:
            runs_by_agent[str(agent)].append(record)

    trades_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    trades_by_agent_date: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    today = datetime.now(timezone.utc).date().isoformat()
    for trade in trades:
        agent = trade.get("agent")
        if agent:
            agent_key = str(agent)
            trades_by_agent[agent_key].append(trade)
        dt = _parse_iso(trade.get("timestamp"))
        if dt is not None and agent:
            trades_by_agent_date[agent_key][dt.date().isoformat()] += 1

    latest_snapshots = _group_latest_snapshots(runs)
    active_markets = _count_markets(_load_json(markets_path))

    agent_entries: List[Dict[str, Any]] = []
    for cfg in agent_configs:
        agent_runs = runs_by_agent.get(cfg.name, [])
        trade_entries = trades_by_agent.get(cfg.name, [])
        trades_today = sum(
            1
            for entry in trade_entries
            if (_parse_iso(entry.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)).date().isoformat()
            == today
        )
        history = _build_history(agent_runs, defaultdict(int, trades_by_agent_date.get(cfg.name, {})))
        latest_snapshot = latest_snapshots.get(cfg.name, {}).get("snapshot", {})
        provider_key = cfg.model_provider.lower()
        color, color_muted = PROVIDER_COLORS.get(provider_key, ("#e2e8f0", "rgba(226, 232, 240, 0.18)"))
        bankroll = float(latest_snapshot.get("bankroll") or 0.0)
        positions_value = float(latest_snapshot.get("positions_value") or 0.0)
        daily_pnl = history[-1]["pnl"] if history else 0.0
        agent_entries.append(
            {
                "name": cfg.name,
                "slug": _slugify(cfg.name),
                "model": cfg.model,
                "provider": PROVIDER_LABELS.get(provider_key, cfg.model_provider),
                "wallet": latest_snapshot.get("wallet") or "",
                "cash": float(latest_snapshot.get("cash_balance") or 0.0),
                "bankroll": bankroll,
                "totalAssets": bankroll,
                "positionsValue": positions_value,
                "dailyPnl": daily_pnl,
                "openPositions": int(latest_snapshot.get("open_positions") or 0),
                "winRate": 0.0,
                "color": color,
                "colorMuted": color_muted,
                "notes": "Auto-synced from daily Agent-Time runs.",
                "trades": _build_trades(trade_entries, cfg.name, agent_runs),
                "history": history,
                "tradesToday": trades_today,
                "positions": latest_snapshot.get("positions") or [],
            }
        )

    summary = {
        "activeMarkets": active_markets,
        "totalTradesToday": sum(entry["tradesToday"] for entry in agent_entries),
        "manaInPlay": _sum_mana_in_play(trades),
    }

    return {
        "lastUpdated": _latest_timestamp(runs, trades),
        "summary": summary,
        "agents": agent_entries,
    }


def main() -> None:
    args = _parse_args()
    agents_path = Path(args.agents)
    results_path = Path(args.results)
    trades_path = Path(args.trades)
    markets_path = Path(args.markets)
    output_path = Path(args.out)

    payload = build_payload(
        agents_path=agents_path,
        results_path=results_path,
        trades_path=trades_path,
        markets_path=markets_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
