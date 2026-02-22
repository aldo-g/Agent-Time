#!/usr/bin/env python3
"""Serve the Predict Arena UI plus a live Manifold-backed API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import parse_qs, urlparse

import utils.env_loader as env_loader  # noqa: F401
from agent.manifold.portfolio import (
    PortfolioSnapshot,
    fetch_portfolio_snapshot,
    fetch_user_overview,
)
from agent.web.export_dashboard import build_payload

logger = logging.getLogger(__name__)
_LIVE_RUNS_CACHE: Dict[str, Any] = {"payload": None, "ts": None}
_LIVE_RUNS_TTL_SECONDS = 3600

def _mask_key(key: str) -> str:
    return key if len(key) <= 8 else f"{key[:4]}...{key[-4:]}"


def _fetch_live_snapshot(name: str, key: str, source_label: str, debug: bool) -> tuple[PortfolioSnapshot, Optional[dict]]:
    logger.info(
        "Fetching live Manifold positions for %s using %s=%s.",
        name,
        source_label,
        _mask_key(key),
    )
    snapshot = fetch_portfolio_snapshot(api_key=key)
    me_overview = fetch_user_overview(api_key=key) if debug else None
    return snapshot, me_overview


def _load_agents(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _resolve_manifold_key(agent: dict) -> Optional[str]:
    if agent.get("manifold_key"):
        return str(agent["manifold_key"])
    env_key = agent.get("manifold_key_env")
    if env_key:
        return os.environ.get(str(env_key))
    return None


def _resolve_expected_wallet(agent: dict) -> Optional[str]:
    expected = agent.get("expected_wallet") or agent.get("wallet")
    if expected:
        return str(expected)
    return None


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = float(snapshot.unrealized_pnl)
    if positions_value is not None:
        positions_value = float(positions_value)
        return cash + positions_value, abs(positions_value)
    net_value = 0.0
    gross_exposure = 0.0
    for position in snapshot.positions:
        value = position.estimated_value()
        if value is None:
            continue
        net_value += value
        gross_exposure += abs(value)
    return cash + net_value, gross_exposure


def _snapshot_to_dict(snapshot: PortfolioSnapshot) -> Dict[str, Any]:
    bankroll, gross_exposure = _estimate_bankroll(snapshot)
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = float(snapshot.unrealized_pnl)
    if positions_value is None:
        positions_value = 0.0
        for position in snapshot.positions:
            value = position.estimated_value()
            if value is None:
                continue
            positions_value += value
    positions = [
        {
            "market_id": position.market_id,
            "question": position.question,
            "outcome": position.outcome,
            "shares": position.shares,
            "avg_price": position.avg_price,
            "mark_price": position.mark_price,
            "pnl": position.pnl,
        }
        for position in snapshot.positions
    ]
    return {
        "wallet": snapshot.wallet,
        "cash_balance": snapshot.cash_balance,
        "realized_pnl": snapshot.realized_pnl,
        "unrealized_pnl": snapshot.unrealized_pnl,
        "investment_value": snapshot.investment_value,
        "cash_investment_value": snapshot.cash_investment_value,
        "positions_value": positions_value,
        "bankroll": bankroll,
        "gross_exposure": gross_exposure,
        "open_positions": len(snapshot.positions),
        "positions": positions,
    }


def _retry_on_wallet_mismatch(
    *,
    name: str,
    key: str,
    source_label: str,
    expected_wallet: str,
    debug: bool,
    max_attempts: int = 3,
    delay_seconds: float = 0.6,
) -> tuple[PortfolioSnapshot, Optional[dict], str]:
    """Retry fetching when Manifold briefly returns a stale wallet mapping."""

    snapshot: PortfolioSnapshot
    me_overview: Optional[dict]
    live_wallet = ""
    for attempt in range(max_attempts):
        snapshot, me_overview = _fetch_live_snapshot(name, key, source_label, debug)
        live_wallet = str(snapshot.wallet or "")
        if live_wallet.lower() == expected_wallet.lower():
            break
        if attempt < max_attempts - 1:
            logger.warning(
                "Wallet mismatch for %s: expected %s, got %s. Retrying (%d/%d)...",
                name,
                expected_wallet,
                live_wallet or "unknown",
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay_seconds)
    return snapshot, me_overview, live_wallet


def _hydrate_live_positions(
    payload: Dict[str, Any],
    agents: Iterable[dict],
    *,
    debug: bool = False,
) -> None:
    agent_map = {agent.get("name"): agent for agent in agents if agent.get("name")}
    work_items: list[tuple[dict, dict, str, str]] = []
    for agent_entry in payload.get("agents", []):
        name = agent_entry.get("name")
        if not name:
            continue
        config = agent_map.get(name)
        if not config:
            logger.warning("No config found for agent %s; skipping live hydration.", name)
            agent_entry["liveHydration"] = {"status": "skipped", "reason": "missing_config"}
            continue
        key = _resolve_manifold_key(config)
        if not key:
            logger.warning(
                "Missing Manifold API key for agent %s (env %s).",
                name,
                config.get("manifold_key_env") or "manifold_key",
            )
            agent_entry["liveHydration"] = {
                "status": "skipped",
                "reason": "missing_key",
                "env": config.get("manifold_key_env") or "manifold_key",
            }
            continue
        source_label = config.get("manifold_key_env") or "manifold_key"
        work_items.append((agent_entry, config, key, source_label))

    if not work_items:
        payload["lastUpdated"] = datetime.now(timezone.utc).isoformat()
        return

    max_workers = min(4, len(work_items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(_fetch_live_snapshot, config.get("name") or "", key, source_label, debug): (agent_entry, config)
            for agent_entry, config, key, source_label in work_items
        }
        for future in as_completed(future_to_item):
            agent_entry, config = future_to_item[future]
            name = config.get("name") or agent_entry.get("name") or "unknown"
            try:
                snapshot, me_overview = future.result()
            except Exception as exc:
                logger.warning("Live Manifold fetch failed for %s: %s", name, exc)
                agent_entry["liveHydration"] = {"status": "error", "reason": str(exc)}
                continue
            expected_wallet = _resolve_expected_wallet(config)
            if expected_wallet:
                live_wallet = str(snapshot.wallet or "")
                if live_wallet.lower() != expected_wallet.lower():
                    snapshot, me_overview, live_wallet = _retry_on_wallet_mismatch(
                        name=name,
                        key=_resolve_manifold_key(config) or "",
                        source_label=config.get("manifold_key_env") or "manifold_key",
                        expected_wallet=expected_wallet,
                        debug=debug,
                    )
                if live_wallet.lower() != expected_wallet.lower():
                    logger.warning(
                        "Wallet mismatch for %s: expected %s, got %s.",
                        name,
                        expected_wallet,
                        live_wallet or "unknown",
                    )
                    agent_entry["liveHydration"] = {
                        "status": "mismatch",
                        "expected": expected_wallet,
                        "found": live_wallet,
                    }
                    continue
            live = _snapshot_to_dict(snapshot)
            agent_entry["wallet"] = live.get("wallet", agent_entry.get("wallet", ""))
            agent_entry["cash"] = float(live.get("cash_balance") or 0.0)
            agent_entry["bankroll"] = float(live.get("bankroll") or 0.0)
            agent_entry["totalAssets"] = agent_entry["bankroll"]
            agent_entry["positionsValue"] = float(live.get("positions_value") or 0.0)
            agent_entry["openPositions"] = int(live.get("open_positions") or 0)
            agent_entry["positions"] = live.get("positions") or []
            hydration = {"status": "ok", "positions": len(agent_entry["positions"])}
            if debug and me_overview is not None:
                hydration["me"] = me_overview
            agent_entry["liveHydration"] = hydration
    payload["lastUpdated"] = datetime.now(timezone.utc).isoformat()


class PredictArenaHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, static_dir: Path, api_config: dict, **kwargs: Any) -> None:
        self.static_dir = static_dir
        self.api_config = api_config
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._handle_api()
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _handle_api(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        bypass_cache = query.get("refresh") == ["1"] or query.get("cache") == ["0"]
        if path.startswith("/api/health"):
            self._send_json({"status": "ok"})
            return
        if path.startswith("/api/live-runs"):
            now = datetime.now(timezone.utc)
            cached_payload = _LIVE_RUNS_CACHE.get("payload")
            cached_ts = _LIVE_RUNS_CACHE.get("ts")
            if (
                not bypass_cache
                and cached_payload
                and cached_ts
                and (now - cached_ts).total_seconds() < _LIVE_RUNS_TTL_SECONDS
            ):
                self._send_json(cached_payload, cache_seconds=_LIVE_RUNS_TTL_SECONDS)
                return
            payload = build_payload(
                agents_path=self.api_config["agents_path"],
                results_path=self.api_config["results_path"],
                trades_path=self.api_config["trades_path"],
                markets_path=self.api_config["markets_path"],
            )
            _hydrate_live_positions(payload, self.api_config["agents"], debug=bypass_cache and query.get("debug") == ["1"])
            _LIVE_RUNS_CACHE["payload"] = payload
            _LIVE_RUNS_CACHE["ts"] = now
            self._send_json(payload, cache_seconds=_LIVE_RUNS_TTL_SECONDS)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(
        self,
        payload: object,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache_seconds: int | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cache_seconds is None:
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", f"public, max-age={cache_seconds}")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:  # noqa: N802
        if path.endswith(".js"):
            return "application/javascript"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind the server.")
    parser.add_argument("--static-dir", default="web/predict-arena", help="Directory to serve the frontend from.")
    parser.add_argument("--agents", default="agents.json", help="Path to agents.json.")
    parser.add_argument("--results", default="results/gpt_runs.jsonl", help="Path to run logs.")
    parser.add_argument("--trades", default="results/trades.jsonl", help="Path to trade logs.")
    parser.add_argument("--markets", default="data/shared_markets.json", help="Path to market cache.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    args = _parse_args()
    static_dir = Path(args.static_dir).resolve()
    agents_path = Path(args.agents)
    results_path = Path(args.results)
    trades_path = Path(args.trades)
    markets_path = Path(args.markets)
    agents = _load_agents(agents_path)

    api_config = {
        "agents": agents,
        "agents_path": agents_path,
        "results_path": results_path,
        "trades_path": trades_path,
        "markets_path": markets_path,
    }

    handler = lambda *handler_args, **handler_kwargs: PredictArenaHandler(  # noqa: E731
        *handler_args,
        static_dir=static_dir,
        api_config=api_config,
        **handler_kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Predict Arena running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
