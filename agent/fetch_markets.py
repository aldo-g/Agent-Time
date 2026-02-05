#!/usr/bin/env python3
"""Fetch shared Manifold markets and write a cache file."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agent.manifold.data import events_to_dicts, load_open_markets
from agent.tools.manifold.config import MARKET_CACHE_ENV


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
    events = load_open_markets(args.market_limit, 0)
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "limit": args.market_limit,
        "events": events_to_dicts(events),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle)
    print(f"Shared market snapshot saved to {cache_path} ({len(snapshot['events'])} markets).")


if __name__ == "__main__":
    main()
