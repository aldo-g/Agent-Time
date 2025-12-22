#!/usr/bin/env python3
"""Helpers for retrieving a Manifold portfolio snapshot."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import utils.env_loader as env_loader  # noqa: F401
from agent.manifold.constants import MANIFOLD_API_ROOT, MAX_API_LIMIT
USER_AGENT = "AgentTimeBot/1.0 (+https://manifold.markets)"
DEFAULT_BETS_LIMIT = int(os.environ.get("MANIFOLD_BETS_LIMIT", "500"))


def _safe_float(value: object, *, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class PortfolioPosition:
    """Single Manifold position."""

    market_id: str
    question: str
    outcome: str
    shares: float
    slug: Optional[str] = None
    avg_price: Optional[float] = None
    mark_price: Optional[float] = None
    pnl: Optional[float] = None

    def estimated_value(self) -> Optional[float]:
        price = self.mark_price if self.mark_price is not None else self.avg_price
        if price is None:
            return None
        return price * self.shares


@dataclass
class PortfolioSnapshot:
    """Summary of a Manifold account's current exposure."""

    wallet: str
    positions: List[PortfolioPosition] = field(default_factory=list)
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    cash_balance: Optional[float] = None


def fetch_portfolio_snapshot(_: str | None = None) -> PortfolioSnapshot:
    """Fetch the authenticated Manifold user's portfolio."""
    user = _fetch_authenticated_user()
    user_id = str(user.get("id") or user.get("_id") or "")
    if not user_id:
        raise RuntimeError("Unable to determine Manifold user id from /me response.")
    username = user.get("username") or user.get("name") or user_id
    snapshot = PortfolioSnapshot(
        wallet=str(username),
        cash_balance=_safe_float(user.get("balance")),
        realized_pnl=_safe_float(user.get("profitCached")),
        unrealized_pnl=_safe_float(user.get("investmentValue")),
    )
    bets = _fetch_user_bets(user_id, limit=DEFAULT_BETS_LIMIT)
    market_map = _fetch_markets_for_bets(bets)
    snapshot.positions.extend(_build_positions(bets, market_map))
    return snapshot


def _auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("MANIFOLD_API_KEY")
    if not api_key:
        raise RuntimeError("Set MANIFOLD_API_KEY before inspecting the portfolio or trading.")
    return {
        "Authorization": f"Key {api_key}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _api_request(
    path: str,
    *,
    params: dict | None = None,
    method: str = "GET",
    body: object | None = None,
) -> object:
    url = _build_url(path, params)
    headers = _auth_headers()
    data_bytes = None
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data_bytes = bytes(body)
        else:
            data_text = json.dumps(body)
            data_bytes = data_text.encode("utf-8")
    request = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
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


def _build_url(path: str, params: dict | None) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    query = urllib.parse.urlencode(params or {})
    url = f"{MANIFOLD_API_ROOT}{normalized}"
    if query:
        url = f"{url}?{query}"
    return url


def _fetch_authenticated_user() -> dict:
    payload = _api_request("/me")
    if isinstance(payload, dict):
        user = payload.get("user")
        if isinstance(user, dict):
            return user
        return payload
    raise RuntimeError("Unexpected response from Manifold /me endpoint.")


def _fetch_user_bets(user_id: str, *, limit: int) -> List[dict]:
    normalized_limit = min(max(limit, 1), MAX_API_LIMIT)
    params = {
        "userId": user_id,
        "limit": normalized_limit,
    }
    payload = _api_request("/bets", params=params)
    if isinstance(payload, list):
        return [bet for bet in payload if isinstance(bet, dict)]
    if isinstance(payload, dict):
        bets = payload.get("data") or payload.get("bets") or []
        return [bet for bet in bets if isinstance(bet, dict)]
    return []


def _fetch_markets_for_bets(bets: Iterable[dict]) -> Dict[str, dict]:
    unique_ids: List[str] = []
    seen: set[str] = set()
    for bet in bets:
        market_id = bet.get("contractId")
        if not isinstance(market_id, str):
            continue
        if market_id in seen:
            continue
        seen.add(market_id)
        unique_ids.append(market_id)
    market_map: Dict[str, dict] = {}
    for market_id in unique_ids:
        try:
            payload = _api_request(f"/market/{urllib.parse.quote(market_id, safe='')}")
        except urllib.error.HTTPError:
            continue
        except urllib.error.URLError:
            continue
        if isinstance(payload, dict):
            market_map[market_id] = payload
    return market_map


def _build_positions(bets: Iterable[dict], markets: Dict[str, dict]) -> List[PortfolioPosition]:
    aggregates: Dict[Tuple[str, str], Dict[str, float | str | dict | None]] = {}
    for bet in bets:
        contract_id = bet.get("contractId")
        if not isinstance(contract_id, str):
            continue
        market = markets.get(contract_id)
        if not market or market.get("isResolved"):
            continue
        outcome_name = str(bet.get("outcome") or bet.get("answer") or "YES")
        answer_id = bet.get("answerId")
        key = (contract_id, str(answer_id or outcome_name))
        shares = _safe_float(bet.get("shares"), default=0.0) or 0.0
        amount = _safe_float(bet.get("amount"), default=0.0) or 0.0
        direction = -1.0 if amount < 0 else 1.0
        shares_delta = direction * shares
        if abs(shares_delta) < 1e-9:
            continue
        agg = aggregates.setdefault(
            key,
            {
                "shares": 0.0,
                "buy_shares": 0.0,
                "buy_notional": 0.0,
                "question": str(market.get("question") or "Untitled market"),
                "slug": market.get("slug"),
                "outcome_label": _describe_outcome(market, outcome_name, answer_id),
                "market": market,
            },
        )
        agg["shares"] = float(agg["shares"]) + shares_delta
        agg["buy_shares"] = float(agg["buy_shares"])
        agg["buy_notional"] = float(agg["buy_notional"])
        if amount > 0:
            agg["buy_shares"] += shares
            agg["buy_notional"] += amount
    positions: List[PortfolioPosition] = []
    for (market_id, _), agg in aggregates.items():
        shares = float(agg["shares"])
        if abs(shares) < 1e-6:
            continue
        buy_shares = float(agg["buy_shares"])
        avg_price = None
        if buy_shares > 0:
            avg_price = float(agg["buy_notional"]) / buy_shares
        outcome_label = str(agg["outcome_label"])
        market = agg["market"]
        mark = _mark_price(market, outcome_label)
        pnl = None
        if avg_price is not None and mark is not None:
            pnl = (mark - avg_price) * shares
        positions.append(
            PortfolioPosition(
                market_id=market_id,
                question=str(agg["question"]),
                outcome=outcome_label,
                shares=shares,
                slug=str(agg.get("slug") or "") or None,
                avg_price=avg_price,
                mark_price=mark,
                pnl=pnl,
            )
        )
    positions.sort(key=lambda pos: abs(pos.estimated_value() or 0.0), reverse=True)
    return positions


def _describe_outcome(market: dict, outcome: str, answer_id: object) -> str:
    outcome_type = str(market.get("outcomeType") or "").upper()
    outcome_upper = (outcome or "").strip().upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"} and outcome_upper in {"YES", "NO"}:
        return outcome_upper
    if answer_id:
        for answer in market.get("answers") or []:
            if isinstance(answer, dict) and answer.get("id") == answer_id:
                text = answer.get("text")
                if text:
                    return str(text)
    return outcome or "Unknown"


def _mark_price(market: dict, outcome_label: str) -> Optional[float]:
    outcome_type = str(market.get("outcomeType") or "").upper()
    probability = _safe_float(market.get("probability"))
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"} and probability is not None:
        prob = max(min(probability, 1.0), 0.0)
        if outcome_label.strip().upper() == "YES":
            return prob
        if outcome_label.strip().upper() == "NO":
            return 1.0 - prob
    answers = market.get("answers") or []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        text = str(answer.get("text") or "").strip()
        if text and text.lower() == outcome_label.strip().lower():
            return _safe_float(answer.get("probability"))
    return probability


__all__ = ["PortfolioPosition", "PortfolioSnapshot", "fetch_portfolio_snapshot"]


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body or "no response body"
