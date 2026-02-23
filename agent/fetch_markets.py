#!/usr/bin/env python3
"""Fetch shared Manifold markets and write a cache file."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.db import DbWriter
from agent.manifold.data import events_to_dicts, load_open_markets
from agent.tools.manifold.config import MARKET_CACHE_ENV


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _connect_db_with_retry() -> DbWriter:
    attempts = max(1, _env_int("AGENT_DB_CONNECT_RETRIES", 30))
    delay_seconds = max(0.1, _env_float("AGENT_DB_CONNECT_RETRY_DELAY_SECONDS", 2.0))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            db_writer = DbWriter.from_env()
            db_writer.ping()
            db_writer.ensure_schema()
            return db_writer
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                print(
                    f"Database not ready yet (attempt {attempt}/{attempts}): {exc}. "
                    f"Retrying in {delay_seconds:.1f}s..."
                )
                time.sleep(delay_seconds)
    assert last_error is not None
    raise RuntimeError(last_error)


def _load_agents_config(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Agent config file must contain a JSON object or a JSON list.")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch shared markets and cache them to disk.")
    parser.add_argument(
        "--market-limit",
        type=int,
        default=int(os.environ.get("AGENT_MARKET_CACHE_LIMIT", "25")),
        help="Number of markets to fetch and cache.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.environ.get(MARKET_CACHE_ENV, "data/shared_markets.json"),
        help="Where to write the shared market cache JSON.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cache_path = Path(args.cache_path)
    fetched_at = datetime.now(timezone.utc)
    events = load_open_markets(args.market_limit, 0)
    snapshot = {
        "fetched_at": fetched_at.isoformat(),
        "limit": args.market_limit,
        "events": events_to_dicts(events),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle)
    print(f"Shared market snapshot saved to {cache_path} ({len(snapshot['events'])} markets).")

    if os.environ.get("DATABASE_URL"):
        try:
            db_writer = _connect_db_with_retry()
            agents_path = Path(os.environ.get("AGENT_CONFIG_PATH", "agent.json"))
            agents = _load_agents_config(agents_path)
            if len(agents) != 1:
                raise ValueError(
                    f"Single-agent mode expects exactly 1 agent entry in {agents_path}; found {len(agents)}."
                )
            session_id = db_writer.create_session(
                started_at=fetched_at,
                market_json=snapshot,
                notes=None,
            )
            agent = agents[0]
            name = agent.get("name")
            model_provider = agent.get("model_provider")
            model = agent.get("model")
            if not name or not model_provider or not model:
                raise ValueError("Agent config entry missing required fields.")
            db_writer.create_run_placeholder(
                session_id=session_id,
                model_provider=str(model_provider),
                model=str(model),
                started_at=fetched_at,
            )
            print(f"Session {session_id} created with 1 run.")
        except Exception as exc:  # noqa: BLE001
            print(f"Database configured but unavailable. Failing market-fetcher. ({exc})")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
