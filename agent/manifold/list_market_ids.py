"""Simple CLI to list all Manifold market ids."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from agent.manifold.constants import MANIFOLD_API_ROOT, MAX_API_LIMIT

USER_AGENT = "AgentTimeBot/1.0 (+https://manifold.markets)"
API_TIMEOUT_SECONDS = 20


def _request_markets(limit: int, before: Optional[int], api_key: Optional[str]) -> Tuple[List[dict], Optional[int]]:
    params: dict[str, object] = {
        "limit": limit,
        "sort": "created-time",
    }
    if before is not None:
        params["before"] = before
    query = urllib.parse.urlencode(params)
    url = f"{MANIFOLD_API_ROOT}/markets?{query}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Key {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url=url,
                    code=response.status,
                    msg=response.reason,
                    hdrs=response.headers,
                    fp=response,
                )
            payload = json.load(response)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = _read_error_body(exc)
        raise RuntimeError(f"Manifold API request failed ({exc.code} {exc.reason}): {detail}") from exc
    if not isinstance(payload, list):
        return [], None
    if not payload:
        return [], None
    created_times = []
    for market in payload:
        if isinstance(market, dict):
            ts = market.get("createdTime")
            try:
                created_times.append(int(ts))
            except (TypeError, ValueError):
                continue
    next_before = min(created_times) - 1 if created_times else None
    return payload, next_before


def list_all_market_ids(api_key: Optional[str], limit: int = MAX_API_LIMIT, max_pages: Optional[int] = None) -> List[str]:
    """Iterate through all markets and return their ids."""
    normalized_limit = min(max(limit, 1), MAX_API_LIMIT)
    before: Optional[int] = None
    ids: List[str] = []
    pages_fetched = 0
    seen = set()
    while True:
        if max_pages is not None and pages_fetched >= max_pages:
            break
        markets, before = _request_markets(normalized_limit, before, api_key)
        if not markets:
            break
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_id = market.get("id") or market.get("_id")
            if not market_id:
                continue
            market_id_str = str(market_id)
            if market_id_str not in seen:
                ids.append(market_id_str)
                seen.add(market_id_str)
        pages_fetched += 1
        if before is None:
            break
        time.sleep(0.2)  # polite pacing to avoid rate limits
    return ids


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body or "no response body"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Manifold market ids.")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Manifold API key (defaults to MANIFOLD_API_KEY env var).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_API_LIMIT,
        help=f"Per-page limit (1-{MAX_API_LIMIT}); default {MAX_API_LIMIT}.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on pages to fetch (omit to fetch all available pages).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    api_key = args.api_key or os.environ.get("MANIFOLD_API_KEY") or None
    try:
        ids = list_all_market_ids(api_key=api_key, limit=args.limit, max_pages=args.max_pages)
    except Exception as exc:  # pragma: no cover - CLI convenience
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    for market_id in ids:
        print(market_id)
    sys.stderr.write(f"Fetched {len(ids)} market ids.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
