"""CLI helper to fetch Manifold markets and print the scout's reasoning."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from agents.market_scout import MarketScout
from connectors.manifold import ManifoldClient, ManifoldAPIError


def fetch_markets(client: ManifoldClient, limit: int, sort: str) -> List[Dict[str, Any]]:
    params = {"limit": limit, "sort": sort}
    return client.list_markets(**params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scout Manifold markets and describe opportunities.")
    parser.add_argument("--limit", type=int, default=40, help="Number of markets to pull (default: 40)")
    parser.add_argument(
        "--sort",
        type=str,
        default="last-bet-time",
        help="Sort order accepted by Manifold (created-time, updated-time, last-bet-time, last-comment-time)",
    )
    args = parser.parse_args(argv)

    client = ManifoldClient()
    try:
        markets = fetch_markets(client, limit=args.limit, sort=args.sort)
    except ManifoldAPIError as exc:
        print(f"Failed to reach Manifold API: {exc}", file=sys.stderr)
        return 1
    scout = MarketScout()
    report = scout.analyze(markets)
    print(report.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
