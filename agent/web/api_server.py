#!/usr/bin/env python3
"""Serve the Predict Arena UI plus a live Manifold-backed API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import utils.env_loader as env_loader  # noqa: F401
from agent.manifold.portfolio import PortfolioSnapshot, fetch_portfolio_snapshot
from agent.web.export_dashboard import build_payload

logger = logging.getLogger(__name__)

@contextmanager
def _temporary_env(var_name: str, value: str):
    previous = os.environ.get(var_name)
    os.environ[var_name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = previous


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


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
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
        "bankroll": bankroll,
        "gross_exposure": gross_exposure,
        "open_positions": len(snapshot.positions),
        "positions": positions,
    }


def _hydrate_live_positions(
    payload: Dict[str, Any],
    agents: Iterable[dict],
) -> None:
    agent_map = {agent.get("name"): agent for agent in agents if agent.get("name")}
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
        try:
            with _temporary_env("MANIFOLD_API_KEY", key):
                logger.info("Fetching live Manifold positions for %s.", name)
                snapshot = fetch_portfolio_snapshot(None)
        except Exception as exc:
            logger.warning("Live Manifold fetch failed for %s: %s", name, exc)
            agent_entry["liveHydration"] = {"status": "error", "reason": str(exc)}
            continue
        live = _snapshot_to_dict(snapshot)
        agent_entry["wallet"] = live.get("wallet", agent_entry.get("wallet", ""))
        agent_entry["cash"] = float(live.get("cash_balance") or 0.0)
        agent_entry["bankroll"] = float(live.get("bankroll") or 0.0)
        agent_entry["totalAssets"] = agent_entry["bankroll"]
        agent_entry["openPositions"] = int(live.get("open_positions") or 0)
        agent_entry["positions"] = live.get("positions") or []
        agent_entry["liveHydration"] = {"status": "ok", "positions": len(agent_entry["positions"])}
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
        if self.path.startswith("/api/health"):
            self._send_json({"status": "ok"})
            return
        if self.path.startswith("/api/live-runs"):
            payload = build_payload(
                agents_path=self.api_config["agents_path"],
                results_path=self.api_config["results_path"],
                trades_path=self.api_config["trades_path"],
                markets_path=self.api_config["markets_path"],
            )
            _hydrate_live_positions(payload, self.api_config["agents"])
            self._send_json(payload)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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
    parser.add_argument("--results", default="results/multi_agent_runs.jsonl", help="Path to run logs.")
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
