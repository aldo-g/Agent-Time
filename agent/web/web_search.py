#!/usr/bin/env python3
"""Lightweight DuckDuckGo web search helper for research workflows."""

from __future__ import annotations

import argparse
import calendar
import os
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional

import utils.env_loader as env_loader  # noqa: F401

try:  # pragma: no cover - optional dependency
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        DDGS = None  # type: ignore[assignment]

DEFAULT_SEARCH_LIMIT = 5


class WebSearchUnavailable(RuntimeError):
    """Raised when the duckduckgo_search dependency is missing."""


@dataclass
class SearchResult:
    """Single search result snippet."""

    title: str
    url: str
    snippet: str


def _resolve_timelimit() -> Optional[str]:
    explicit = os.environ.get("DDG_TIMELIMIT")
    if explicit is not None:
        explicit = explicit.strip()
        return explicit or None
    months = os.environ.get("DDG_RECENT_MONTHS", "6")
    if months is None:
        return None
    try:
        months_back = int(months)
    except ValueError:
        return None
    if months_back <= 0:
        return None
    today = date.today()
    year = today.year
    month = today.month - months_back
    day = today.day
    while month <= 0:
        month += 12
        year -= 1
    days_in_month = calendar.monthrange(year, month)[1]
    if day > days_in_month:
        day = days_in_month
    start = date(year, month, day)
    return f"{start.isoformat()}..{today.isoformat()}"


def search_web(query: str, *, max_results: int = DEFAULT_SEARCH_LIMIT, region: str = "wt-wt") -> List[SearchResult]:
    """Return DuckDuckGo text search results for the query."""
    query = (query or "").strip()
    if not query:
        return []
    if DDGS is None:
        raise WebSearchUnavailable(
            "Install the `ddgs` package to enable web search. `pip install ddgs`."
        )
    max_results = max(1, min(max_results, 25))
    timelimit = _resolve_timelimit()
    with DDGS(timeout=10) as ddgs:
        raw_results: Iterable[dict] = ddgs.text(
            query,
            region=region,
            safesearch="moderate",
            timelimit=timelimit,
            max_results=max_results,
        )
        normalized: List[SearchResult] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title") or entry.get("heading") or "Untitled result")
            url = str(entry.get("href") or entry.get("url") or "")
            snippet = str(entry.get("body") or entry.get("snippet") or "")
            normalized.append(SearchResult(title=title, url=url, snippet=snippet))
        return normalized


def _print_results(results: List[SearchResult]) -> None:
    if not results:
        print("No results.")
        return
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result.title}")
        if result.url:
            print(f"   {result.url}")
        if result.snippet:
            print(f"   {result.snippet}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query text to search for")
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to show (default: 5)")
    args = parser.parse_args()
    try:
        results = search_web(args.query, max_results=args.limit)
    except WebSearchUnavailable as exc:
        print(exc)
        return
    _print_results(results)


if __name__ == "__main__":
    main()
