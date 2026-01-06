"""Helpers for fetching recent Manifold market activity."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from agent.manifold.constants import MANIFOLD_API_ROOT, MAX_API_LIMIT

USER_AGENT = "AgentTimeBot/1.0 (+https://manifold.markets)"


@dataclass
class MarketBet:
    """Summary of a Manifold bet event."""

    timestamp: int
    outcome: str
    amount: float
    prob_after: Optional[float]


def fetch_market_history(market_id: str, limit: int = 200) -> List[MarketBet]:
    """Fetch recent bets for a market."""
    normalized_limit = min(max(limit, 1), MAX_API_LIMIT)
    params = {
        "contractId": market_id,
        "limit": normalized_limit,
    }
    payload = _request("/bets", params=params)
    if not isinstance(payload, list):
        return []
    bets: List[MarketBet] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        timestamp = entry.get("createdTime")
        try:
            created_ms = int(timestamp)
        except (TypeError, ValueError):
            continue
        outcome = str(entry.get("outcome") or entry.get("answer") or "UNKNOWN")
        try:
            amount = float(entry.get("amount") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        try:
            prob_after = float(entry.get("probAfter"))
        except (TypeError, ValueError):
            prob_after = None
        bets.append(MarketBet(timestamp=created_ms, outcome=outcome, amount=amount, prob_after=prob_after))
    bets.sort(key=lambda bet: bet.timestamp, reverse=True)
    return bets


def _request(path: str, *, params: dict | None = None) -> object:
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{MANIFOLD_API_ROOT}{path}?{query}"
    else:
        url = f"{MANIFOLD_API_ROOT}{path}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url=url,
                    code=response.status,
                    msg=response.reason,
                    hdrs=response.headers,
                    fp=response,
                )
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        raise RuntimeError(f"Manifold API request failed ({exc.code} {exc.reason}): {detail}") from exc


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body or "no response body"


__all__ = ["MarketBet", "fetch_market_history"]
