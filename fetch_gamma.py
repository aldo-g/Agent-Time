#!/usr/bin/env python3
"""Fetch current Polymarket events from the public gamma API."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from textwrap import indent, fill

import env_loader  # noqa: F401

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events/pagination"


def fetch_events(limit: int, offset: int) -> dict:
    """Return the JSON payload from the gamma pagination endpoint."""
    params = {
        "limit": str(limit),
        "offset": str(offset),
        "active": "true",
        "archived": "false",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    }
    query = urllib.parse.urlencode(params)
    url = f"{GAMMA_EVENTS_URL}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AgentTimeBot/1.0; +https://polymarket.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"Gamma API returned {response.status} {response.reason}")
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gamma API request failed: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error contacting gamma API: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Number of events to fetch (default: 20)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset in multiples of limit")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the resulting JSON instead of raw output"
    )
    args = parser.parse_args()
    payload = fetch_events(args.limit, args.offset)
    if args.pretty:
        events = None
        if isinstance(payload, dict):
            events = payload.get("events") or payload.get("data")
        elif isinstance(payload, list):
            events = payload
        if not events:
            print("No events returned.")
            return
        for idx, event in enumerate(events, 1):
            event_id = event.get("id", "unknown-id")
            name = event.get("title") or event.get("question") or "Untitled market"
            description = event.get("description") or ""
            print(f"{idx}. {name} [{event_id}]")
            if description:
                print(indent(fill(description, width=100), prefix="    "))
            print()
        return
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
