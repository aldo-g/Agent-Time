"""Shared Manifold data models and helpers used by the LangChain agent."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from agent.manifold.constants import MANIFOLD_API_ROOT, MAX_API_LIMIT, RESOLUTION_CUTOFF_MS

USER_AGENT = "AgentTimeBot/1.0 (+https://manifold.markets)"


@dataclass
class OutcomeQuote:
    name: str
    price: float


@dataclass
class MarketSummary:
    event_id: str
    event_title: str
    market_id: str
    question: str
    url: str | None
    outcomes: List[OutcomeQuote]
    tags: List[str]


@dataclass
class EventSummary:
    event_id: str
    title: str
    url: str | None
    tags: List[str] = field(default_factory=list)
    markets: List[MarketSummary] = field(default_factory=list)


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


def _outcomes_from_market(market: dict) -> List[OutcomeQuote]:
    outcome_type = str(market.get("outcomeType") or "").upper()
    outcomes: List[OutcomeQuote] = []
    probability = market.get("probability")
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        try:
            prob = float(probability)
        except (TypeError, ValueError):
            prob = 0.5
        prob = min(max(prob, 0.0), 1.0)
        outcomes.append(OutcomeQuote(name="YES", price=prob))
        outcomes.append(OutcomeQuote(name="NO", price=1.0 - prob))
        return outcomes
    answers: Sequence[object] = market.get("answers") or []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        name = answer.get("text") or f"Answer {answer.get('index', '-')}"
        try:
            ans_prob = float(answer.get("probability"))
        except (TypeError, ValueError):
            continue
        outcomes.append(OutcomeQuote(name=str(name), price=max(ans_prob, 0.0)))
    if outcomes:
        return outcomes
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        prob = 0.5
    prob = min(max(prob, 0.0), 1.0)
    outcomes.append(OutcomeQuote(name="Top outcome", price=prob))
    return outcomes


def _parse_tags(raw_tags: object) -> List[str]:
    if isinstance(raw_tags, list):
        return [str(tag) for tag in raw_tags if isinstance(tag, str)]
    if isinstance(raw_tags, str):
        return [raw_tags]
    return []


def load_open_markets(limit: int, offset: int) -> List[EventSummary]:
    """Fetch open Manifold markets sorted by recent activity."""
    requested = max(limit + offset, limit)
    api_limit = min(max(requested, 1), MAX_API_LIMIT)
    params = {
        "limit": api_limit,
        "sort": "last-bet-time",
    }
    payload = _request("/markets", params=params)
    market_records: List[dict]
    if isinstance(payload, list):
        market_records = [record for record in payload if isinstance(record, dict)]
    elif isinstance(payload, dict):
        data = payload.get("markets") or payload.get("data") or []
        market_records = [record for record in data if isinstance(record, dict)]
    else:
        market_records = []
    filtered_records = []
    for record in market_records:
        close_time = record.get("closeTime")
        if close_time is None:
            continue
        try:
            close_ms = int(close_time)
        except (TypeError, ValueError):
            continue
        if close_ms > RESOLUTION_CUTOFF_MS:
            continue
        filtered_records.append(record)
    markets = (
        filtered_records[offset : offset + limit]
        if offset or len(filtered_records) > limit
        else filtered_records[:limit]
    )

    summaries: List[EventSummary] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_id = str(market.get("id", ""))
        if not market_id:
            continue
        question = market.get("question") or "Untitled market"
        url = market.get("url") or None
        tags = _parse_tags(market.get("groupSlugs"))
        event = EventSummary(event_id=market_id, title=str(question), url=url, tags=tags)
        outcomes = _outcomes_from_market(market)
        event.markets.append(
            MarketSummary(
                event_id=market_id,
                event_title=str(question),
                market_id=market_id,
                question=str(question),
                url=url,
                outcomes=outcomes,
                tags=tags,
            )
        )
        summaries.append(event)
    return summaries


__all__ = [
    "EventSummary",
    "MarketSummary",
    "OutcomeQuote",
    "load_open_markets",
]
