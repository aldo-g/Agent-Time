"""Helpers for fetching market details and placing Polymarket orders."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
except ImportError:  # pragma: no cover - optional dependency
    ClobClient = None  # type: ignore[assignment]
    ApiCreds = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]
    OrderType = None  # type: ignore[assignment]

DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API_ROOT = os.environ.get("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
USER_AGENT = "AgentTimeBot/1.0 (+https://polymarket.com)"


@dataclass
class MarketTokens:
    """Mapping of available tokens/outcomes for a market."""

    market_id: str
    tokens: Dict[str, str]  # outcome name -> token id
    raw: Dict[str, Any]


@dataclass
class OrderReceipt:
    """Summary of a submitted order."""

    order_id: Optional[str]
    side: str
    token_id: str
    shares: float
    price: float
    usd_notional: float
    response: Any


def _require_py_clob() -> None:
    if ClobClient is None or ApiCreds is None or OrderArgs is None or OrderType is None:  # pragma: no cover - import guard
        raise RuntimeError(
            "py-clob-client is required for order execution. Install it with `pip install py-clob-client`."
        )


def _clob_host() -> str:
    host = (
        os.environ.get("POLYMARKET_EXECUTION_HOST")
        or os.environ.get("POLYMARKET_CASH_API_ROOT")
        or os.environ.get("POLYMARKET_API_ROOT")
        or DEFAULT_CLOB_HOST
    )
    return host.rstrip("/")


def _build_clob_client() -> ClobClient:
    _require_py_clob()
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    wallet = os.environ.get("POLYMARKET_WALLET_ADDRESS")
    api_key = os.environ.get("POLYMARKET_API_KEY")
    api_secret = os.environ.get("POLYMARKET_SECRET")
    api_passphrase = os.environ.get("POLYMARKET_PASSPHRASE")
    if not all([private_key, wallet, api_key, api_secret, api_passphrase]):
        raise RuntimeError(
            "Trading requires POLYMARKET_PRIVATE_KEY, POLYMARKET_WALLET_ADDRESS, "
            "POLYMARKET_API_KEY, POLYMARKET_SECRET, and POLYMARKET_PASSPHRASE."
        )
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
    signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))
    client = ClobClient(
        host=_clob_host(),
        key=private_key,
        chain_id=chain_id,
        funder=wallet,
        signature_type=signature_type,
    )
    client.set_api_creds(
        ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
    )
    return client


def fetch_market_tokens(market_id: str) -> MarketTokens:
    """Return the mapping of outcome -> token id for the given market."""
    if not market_id:
        raise ValueError("market_id is required")
    condition_id = _resolve_condition_id(market_id)
    client = _build_clob_client()
    payload = client.get_market(condition_id)
    tokens: Dict[str, str] = {}
    for record in _iter_token_records(payload):
        token_id = record.get("token_id") or record.get("tokenId")
        name = (
            record.get("outcome")
            or record.get("outcomeName")
            or record.get("name")
            or record.get("ticker")
        )
        if token_id and name:
            tokens[str(name).strip()] = str(token_id)
    if not tokens:
        raise RuntimeError(f"Unable to resolve tokens for market {condition_id}")
    return MarketTokens(market_id=condition_id, tokens=tokens, raw=payload)


def place_limit_order(
    *,
    market_id: str,
    token_id: str,
    side: str,
    price: float,
    shares: float,
    order_type: str = "GTC",
) -> OrderReceipt:
    """Submit a signed limit order through the Polymarket CLOB."""
    if shares <= 0:
        raise ValueError("shares must be positive")
    if price <= 0 or price >= 1:
        raise ValueError("price must be between 0 and 1 (exclusive)")
    condition_id = _resolve_condition_id(market_id)
    client = _build_clob_client()
    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    try:
        clob_order_type = OrderType[order_type.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported order_type {order_type}") from exc
    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=shares,
        side=normalized_side,
    )
    order = client.create_order(order_args)
    response = client.post_order(order, orderType=clob_order_type)
    order_id = None
    if isinstance(response, dict):
        order_id = response.get("order_id") or response.get("orderID") or response.get("id")
    usd_notional = price * shares
    return OrderReceipt(
        order_id=order_id,
        side=normalized_side,
        token_id=token_id,
        shares=shares,
        price=price,
        usd_notional=usd_notional,
        response=response,
    )


def find_token_id(payload: Dict[str, Any], outcome_name: str) -> Optional[str]:
    """Best-effort search for the token id matching the given outcome label."""
    if not payload or not outcome_name:
        return None
    target = outcome_name.strip().lower()
    for record in _iter_token_records(payload):
        name = (
            record.get("outcome")
            or record.get("outcomeName")
            or record.get("name")
            or record.get("ticker")
        )
        token_id = record.get("token_id") or record.get("tokenId")
        if not name or not token_id:
            continue
        if str(name).strip().lower() == target:
            return str(token_id)
    return None


def _resolve_condition_id(identifier: str) -> str:
    ident = (identifier or "").strip()
    if not ident:
        raise RuntimeError("market_id is required")
    if ident.startswith("0x"):
        return ident
    market_payload = _fetch_gamma_market_payload(ident)
    if not isinstance(market_payload, dict):
        raise RuntimeError(f"Unable to resolve market payload for {ident}")
    condition = (
        market_payload.get("conditionId")
        or market_payload.get("condition_id")
        or market_payload.get("conditionID")
    )
    if not condition:
        raise RuntimeError(f"Resolved payload for {ident} lacks conditionId.")
    return str(condition)


def _fetch_gamma_market_payload(identifier: str) -> dict:
    if not identifier:
        raise RuntimeError("market identifier is required")
    candidate_paths = [
        f"/markets/{urllib.parse.quote(identifier, safe='')}",
        f"/markets?slug={urllib.parse.quote(identifier, safe='')}",
        f"/markets?ids={urllib.parse.quote(identifier, safe='')}",
    ]
    last_error: Optional[Exception] = None
    for path in candidate_paths:
        try:
            payload = _gamma_request(path)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        for record in _iter_market_records(payload):
            condition = (
                record.get("conditionId")
                or record.get("condition_id")
                or record.get("conditionID")
            )
            if condition:
                return record
    if last_error:
        raise RuntimeError(f"Unable to resolve market {identifier}: {last_error}") from last_error
    raise RuntimeError(f"Unable to resolve market {identifier}: conditionId not found.")


def _gamma_request(path: str) -> object:
    url = f"{GAMMA_API_ROOT.rstrip('/')}/{path.lstrip('/')}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
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


def _iter_market_records(payload: object) -> Iterable[Dict[str, Any]]:
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            keys = {k.lower() for k in item.keys()}
            if any(key in keys for key in ("conditionid", "condition_id", "conditionid")):
                yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _iter_token_records(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield candidate token dicts from arbitrarily nested payloads."""
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if ("token_id" in item or "tokenId" in item) and any(
                key in item for key in ("outcome", "outcomeName", "name", "ticker")
            ):
                yield item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


__all__ = [
    "MarketTokens",
    "OrderReceipt",
    "fetch_market_tokens",
    "place_limit_order",
    "find_token_id",
]
