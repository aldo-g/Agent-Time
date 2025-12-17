"""CLI helper to fetch Polymarket markets and print the scout's reasoning."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from agents.market_scout import MarketScout
from connectors.polymarket import PolymarketClient, PolymarketAPIError


def fetch_markets(client: PolymarketClient, limit: int, sort: str) -> List[Dict[str, Any]]:
    params = {"limit": limit, "sort": sort}
    return client.list_markets(**params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scout Polymarket markets and describe opportunities.")
    parser.add_argument("--limit", type=int, default=40, help="Number of markets to pull (default: 40)")
    parser.add_argument(
        "--sort",
        type=str,
        default="last-bet-time",
        help="Sort order accepted by the Polymarket client (last-bet-time, updated-time, volume24h, liquidity)",
    )
    args = parser.parse_args(argv)

    client = PolymarketClient()
    try:
        markets = fetch_markets(client, limit=args.limit, sort=args.sort)
    except PolymarketAPIError as exc:
        print(f"Failed to reach Polymarket API: {exc}", file=sys.stderr)
        return 1
    scout = MarketScout()
    report = scout.analyze(markets)
    print(report.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
