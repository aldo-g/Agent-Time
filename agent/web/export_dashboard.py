#!/usr/bin/env python3
"""Build the Predict Arena frontend payload from run logs."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.db import DbWriter


PROVIDER_LABELS = {
    "openai": "OpenAI",
}

PROVIDER_COLORS = {
    "openai": ("#3f7f5f", "rgba(63, 127, 95, 0.16)"),
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
                "bankroll": bankroll,
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
                "amount": abs(_coerce_float(entry.get("amount"), default=0.0) or 0.0),
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


def _coerce_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_datetime(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return _parse_iso(value)
    return None


def _coerce_json_dict(value: object) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_json_list(value: object) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _extract_sources(reason: str, raw_response: Dict[str, Any]) -> List[str]:
    sources = _extract_links(reason)
    catalyst_urls = raw_response.get("catalyst_urls")
    if isinstance(catalyst_urls, list):
        for entry in catalyst_urls:
            url = str(entry or "").strip()
            if url:
                sources.append(url)
    deduped: List[str] = []
    seen: set[str] = set()
    for url in sources:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _build_history_from_db(
    *,
    run_rows: List[tuple],
    trades_by_date: Dict[str, int],
) -> List[Dict[str, Any]]:
    latest_by_date: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:
        started_at = _coerce_datetime(row[1])
        finished_at = _coerce_datetime(row[2])
        ts = finished_at or started_at
        if ts is None:
            continue
        bankroll = _coerce_float(row[3], default=float("nan"))
        if bankroll != bankroll:  # NaN guard
            continue
        date_key = ts.date().isoformat()
        existing = latest_by_date.get(date_key)
        if not existing or ts > existing["timestamp"]:
            latest_by_date[date_key] = {"timestamp": ts, "bankroll": bankroll}
    history: List[Dict[str, Any]] = []
    prev_bankroll: Optional[float] = None
    for date_key in sorted(latest_by_date.keys()):
        bankroll = latest_by_date[date_key]["bankroll"]
        pnl = bankroll - prev_bankroll if prev_bankroll is not None else 0.0
        history.append(
            {
                "date": date_key,
                "bankroll": bankroll,
                "pnl": pnl,
                "trades": trades_by_date.get(date_key, 0),
            }
        )
        prev_bankroll = bankroll
    return history


def _build_payload_from_db(*, agents_path: Path, markets_path: Path) -> Dict[str, Any]:
    agents_payload = _load_json(agents_path) or []
    if isinstance(agents_payload, dict):
        agents_payload = [agents_payload]
    agent_configs = [
        AgentConfig.from_dict(entry)
        for entry in agents_payload
        if isinstance(entry, dict) and entry.get("name")
    ]
    if not agent_configs:
        return {
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "summary": {"activeMarkets": 0, "totalTradesToday": 0, "manaInPlay": 0.0},
            "agents": [],
        }

    db_writer = DbWriter.from_env()
    db_writer.ping()
    db_writer.ensure_schema()

    active_markets = 0
    timestamps: List[datetime] = []
    agent_entries: List[Dict[str, Any]] = []
    today = datetime.now(timezone.utc).date().isoformat()

    with db_writer.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT market_json, started_at
                FROM sessions
                ORDER BY started_at DESC, id DESC
                LIMIT 1;
                """
            )
            session_row = cur.fetchone()
            if session_row:
                active_markets = _count_markets(session_row[0])
                ts = _coerce_datetime(session_row[1])
                if ts is not None:
                    timestamps.append(ts)
            if active_markets <= 0:
                active_markets = _count_markets(_load_json(markets_path))

            for cfg in agent_configs:
                provider_key = cfg.model_provider.lower()
                color, color_muted = PROVIDER_COLORS.get(provider_key, ("#e2e8f0", "rgba(226, 232, 240, 0.18)"))
                cur.execute(
                    """
                    SELECT id, started_at, finished_at, bankroll, cash_balance, position_balance, metadata
                    FROM runs
                    WHERE model_provider = %s AND model = %s
                    ORDER BY COALESCE(finished_at, started_at), id;
                    """,
                    (cfg.model_provider, cfg.model),
                )
                run_rows = cur.fetchall()
                run_tools: Dict[int, List[str]] = {}
                for row in run_rows:
                    run_id = int(row[0])
                    metadata = _coerce_json_dict(row[6])
                    tools_raw = metadata.get("tool_calls")
                    run_tools[run_id] = [str(tool) for tool in _coerce_json_list(tools_raw) if str(tool).strip()]
                    started_at = _coerce_datetime(row[1])
                    finished_at = _coerce_datetime(row[2])
                    if started_at is not None:
                        timestamps.append(started_at)
                    if finished_at is not None:
                        timestamps.append(finished_at)

                latest_run_id: Optional[int] = None
                latest_run_time: Optional[datetime] = None
                latest_finished_run_id: Optional[int] = None
                latest_finished_time: Optional[datetime] = None
                for row in run_rows:
                    run_id = int(row[0])
                    started_at = _coerce_datetime(row[1])
                    finished_at = _coerce_datetime(row[2])
                    run_time = finished_at or started_at
                    if run_time is None:
                        continue
                    if latest_run_time is None or run_time > latest_run_time:
                        latest_run_time = run_time
                        latest_run_id = run_id
                    if finished_at is not None and (
                        latest_finished_time is None or finished_at > latest_finished_time
                    ):
                        latest_finished_time = finished_at
                        latest_finished_run_id = run_id
                if latest_finished_run_id is not None:
                    latest_run_id = latest_finished_run_id

                cash_balance = 0.0
                bankroll = 0.0
                positions_value = 0.0
                wallet = ""
                positions: List[Dict[str, Any]] = []
                open_positions = 0

                if latest_run_id is not None:
                    cur.execute(
                        """
                        SELECT cash_balance, positions_value, bankroll, open_positions, snapshot_json
                        FROM equity_snapshots
                        WHERE run_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1;
                        """,
                        (latest_run_id,),
                    )
                    snapshot_row = cur.fetchone()
                    snapshot_json: Dict[str, Any] = {}
                    if snapshot_row:
                        cash_balance = _coerce_float(snapshot_row[0])
                        positions_value = _coerce_float(snapshot_row[1])
                        bankroll = _coerce_float(snapshot_row[2])
                        open_positions = int(snapshot_row[3] or 0)
                        snapshot_json = _coerce_json_dict(snapshot_row[4])
                        wallet = str(snapshot_json.get("wallet") or "")
                    cur.execute(
                        """
                        SELECT
                            market_id,
                            market_slug,
                            question,
                            outcome,
                            shares,
                            avg_price,
                            mark_price,
                            position_value,
                            pnl,
                            created_at
                        FROM open_positions
                        WHERE run_id = %s
                        ORDER BY ABS(position_value) DESC NULLS LAST, id DESC;
                        """,
                        (latest_run_id,),
                    )
                    position_rows = cur.fetchall()
                    for row in position_rows:
                        ts = _coerce_datetime(row[9])
                        if ts is not None:
                            timestamps.append(ts)
                        positions.append(
                            {
                                "market_id": row[0],
                                "market_slug": row[1],
                                "question": row[2],
                                "outcome": row[3],
                                "shares": _coerce_float(row[4], default=0.0),
                                "avg_price": _coerce_float(row[5], default=None),
                                "mark_price": _coerce_float(row[6], default=None),
                                "position_value": _coerce_float(row[7], default=None),
                                "pnl": _coerce_float(row[8], default=None),
                            }
                        )
                    if positions and open_positions <= 0:
                        open_positions = len(positions)
                    if bankroll <= 0 and latest_run_id is not None:
                        for row in reversed(run_rows):
                            if int(row[0]) != latest_run_id:
                                continue
                            bankroll = _coerce_float(row[3], default=0.0)
                            cash_balance = _coerce_float(row[4], default=0.0)
                            positions_value = _coerce_float(row[5], default=0.0)
                            break

                cur.execute(
                    """
                    SELECT
                        te.id,
                        te.run_id,
                        te.market_id,
                        te.market_slug,
                        te.action,
                        te.outcome,
                        te.amount,
                        te.prob_before,
                        te.prob_after,
                        te.status,
                        te.reason,
                        te.error,
                        te.summary,
                        te.bet_id,
                        te.raw_response,
                        te.created_at
                    FROM trade_executions te
                    JOIN runs r ON r.id = te.run_id
                    WHERE r.model_provider = %s AND r.model = %s
                    ORDER BY te.created_at DESC, te.id DESC;
                    """,
                    (cfg.model_provider, cfg.model),
                )
                trade_rows = cur.fetchall()
                trades_by_date: Dict[str, int] = defaultdict(int)
                trades: List[Dict[str, Any]] = []
                trades_today = 0
                for row in trade_rows:
                    trade_id = row[0]
                    run_id = int(row[1])
                    market_id = str(row[2] or "")
                    market_slug = str(row[3] or "")
                    action = str(row[4] or "").upper() or "BUY"
                    outcome = str(row[5] or "")
                    amount = abs(_coerce_float(row[6], default=0.0))
                    prob_before = _coerce_float(row[7], default=None)
                    prob_after = _coerce_float(row[8], default=None)
                    raw_status = str(row[9] or "").strip()
                    status_lower = raw_status.lower()
                    if status_lower == "executed":
                        status = "EXECUTED"
                    elif status_lower == "open":
                        status = "OPEN"
                    elif status_lower == "skipped":
                        status = "SKIPPED"
                    elif status_lower == "failed":
                        status = "FAILED"
                    elif raw_status:
                        status = raw_status.upper()
                    else:
                        status = "OPEN"
                    raw_response = _coerce_json_dict(row[14])
                    market = str(raw_response.get("market") or row[12] or market_id)
                    market_url = str(raw_response.get("market_url") or "")
                    if not market_url and market_slug:
                        market_url = f"https://manifold.markets/{market_slug}"
                    reason = str(row[10] or row[11] or row[12] or "")
                    sources = _extract_sources(reason, raw_response)
                    created_at = _coerce_datetime(row[15])
                    timestamp = (
                        created_at.isoformat()
                        if created_at is not None
                        else datetime.now(timezone.utc).isoformat()
                    )
                    if created_at is not None:
                        timestamps.append(created_at)
                        if created_at.date().isoformat() == today and status not in {"SKIPPED", "FAILED"}:
                            trades_today += 1
                            trades_by_date[created_at.date().isoformat()] += 1
                    trades.append(
                        {
                            "id": str(row[13] or trade_id),
                            "timestamp": timestamp,
                            "market": market,
                            "marketUrl": market_url,
                            "action": action,
                            "outcome": outcome,
                            "amount": amount,
                            "probBefore": prob_before,
                            "probAfter": prob_after,
                            "status": status,
                            "reason": reason,
                            "tools": run_tools.get(run_id, []),
                            "sources": sources,
                        }
                    )

                history = _build_history_from_db(run_rows=run_rows, trades_by_date=trades_by_date)
                daily_pnl = history[-1]["pnl"] if history else 0.0
                agent_entries.append(
                    {
                        "name": cfg.name,
                        "slug": _slugify(cfg.name),
                        "model": cfg.model,
                        "provider": PROVIDER_LABELS.get(provider_key, cfg.model_provider),
                        "wallet": wallet,
                        "cash": cash_balance,
                        "bankroll": bankroll,
                        "totalAssets": bankroll,
                        "positionsValue": positions_value,
                        "dailyPnl": daily_pnl,
                        "openPositions": open_positions,
                        "winRate": 0.0,
                        "color": color,
                        "colorMuted": color_muted,
                        "notes": "Synced from Postgres run/trade/position tables.",
                        "trades": trades,
                        "history": history,
                        "tradesToday": trades_today,
                        "positions": positions,
                    }
                )

    mana_in_play = 0.0
    for entry in agent_entries:
        for position in entry.get("positions", []):
            mana_in_play += abs(_coerce_float(position.get("position_value"), default=0.0))

    last_updated = max(timestamps).isoformat() if timestamps else datetime.now(timezone.utc).isoformat()
    return {
        "lastUpdated": last_updated,
        "summary": {
            "activeMarkets": active_markets,
            "totalTradesToday": sum(entry.get("tradesToday", 0) for entry in agent_entries),
            "manaInPlay": mana_in_play,
        },
        "agents": agent_entries,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="agent.json", help="Path to agent.json.")
    parser.add_argument("--results", default="results/gpt_runs.jsonl", help="JSONL run log path.")
    parser.add_argument("--trades", default="results/trades.jsonl", help="JSONL trade log path.")
    parser.add_argument("--markets", default="data/shared_markets.json", help="Shared market cache path.")
    parser.add_argument(
        "--out",
        default="web/predict-arena/data/live_runs.json",
        help="Output payload path for the frontend.",
    )
    return parser.parse_args()


def _build_payload_from_logs(
    *,
    agents_path: Path,
    results_path: Path,
    trades_path: Path,
    markets_path: Path,
) -> Dict[str, Any]:
    agents_payload = _load_json(agents_path) or []
    if isinstance(agents_payload, dict):
        agents_payload = [agents_payload]
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


def build_payload(
    *,
    agents_path: Path,
    results_path: Path,
    trades_path: Path,
    markets_path: Path,
) -> Dict[str, Any]:
    if os.environ.get("DATABASE_URL"):
        try:
            return _build_payload_from_db(agents_path=agents_path, markets_path=markets_path)
        except Exception:
            # File-based payload remains as a fallback path.
            pass
    return _build_payload_from_logs(
        agents_path=agents_path,
        results_path=results_path,
        trades_path=trades_path,
        markets_path=markets_path,
    )


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
