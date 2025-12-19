#!/usr/bin/env python3
"""Helpers for retrieving a Polymarket portfolio snapshot."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import utils.env_loader as env_loader  # noqa: F401

USER_AGENT = "Mozilla/5.0 (compatible; AgentTimeBot/1.0; +https://polymarket.com)"
DEFAULT_API_ROOT = "https://clob.polymarket.com"
PORTFOLIO_URL_TEMPLATES = [
    "https://clob.polymarket.com/positions?wallet={wallet}",
    "https://clob.polymarket.com/positions/{wallet}",
    "https://gamma-api.polymarket.com/portfolio/{wallet}",
    "https://gamma-api.polymarket.com/portfolio?wallet={wallet}",
]
DATA_API_TEMPLATES = [
    "https://data-api.polymarket.com/positions?user={wallet}",
    "https://data-api.polymarket.com/positions?wallet={wallet}",
    "https://data-api.polymarket.com/positions?address={wallet}",
    (
        "https://data-api.polymarket.com/positions"
        "?user={wallet}&sizeThreshold=0.1&limit=200&offset=0&sortBy=CURRENT&sortDirection=DESC"
    ),
]
AUTH_PORTFOLIO_PATH_TEMPLATES = [
    "/api/v1/portfolio/balances",
    "/api/v1/portfolio/positions",
    "/api/v1/portfolio",
    "/api/v1/portfolio/{wallet}",
    "/api/v1/portfolio?wallet={wallet}",
    "/api/v1/users/me/portfolio",
]


def _safe_float(value: object, *, default: Optional[float] = None) -> Optional[float]:
    """Convert arbitrary JSON values to floats."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class PortfolioPosition:
    """Single Polymarket position."""

    market_id: str
    question: str
    outcome: str
    shares: float
    slug: Optional[str] = None
    avg_price: Optional[float] = None
    mark_price: Optional[float] = None
    pnl: Optional[float] = None

    def estimated_value(self) -> Optional[float]:
        """Return an approximate USD value using mark price then average price."""
        price = self.mark_price if self.mark_price is not None else self.avg_price
        if price is None:
            return None
        return price * self.shares


@dataclass
class PortfolioSnapshot:
    """Summary of a wallet's current Polymarket exposure."""

    wallet: str
    positions: List[PortfolioPosition] = field(default_factory=list)
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    cash_balance: Optional[float] = None


def _portfolio_url_candidates(wallet: str) -> List[str]:
    wallet_variants = _unique_wallet_variants(wallet)
    templates: List[str] = []
    override = os.environ.get("POLYMARKET_PORTFOLIO_URL")
    if override:
        templates.append(override)
    templates.extend(PORTFOLIO_URL_TEMPLATES + DATA_API_TEMPLATES)
    urls: List[str] = []
    for template in templates:
        if not template:
            continue
        for encoded_wallet in wallet_variants:
            if "{wallet}" in template:
                urls.append(template.format(wallet=encoded_wallet))
            else:
                separator = "&" if "?" in template else "?"
                urls.append(f"{template}{separator}wallet={encoded_wallet}")
    return urls


def _auth_portfolio_paths(wallet: str) -> List[str]:
    wallet_variants = _unique_wallet_variants(wallet, encode=False)
    templates: List[str] = []
    override = os.environ.get("POLYMARKET_AUTH_PORTFOLIO_PATH")
    if override:
        templates.append(override)
    templates.extend(AUTH_PORTFOLIO_PATH_TEMPLATES)
    paths: List[str] = []
    seen: set[str] = set()
    for template in templates:
        if not template:
            continue
        if "{wallet}" not in template:
            candidate = template
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
            continue
        for variant in wallet_variants:
            encoded = urllib.parse.quote(variant, safe="0x")
            candidate = template.format(wallet=encoded)
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return paths


def _unique_wallet_variants(wallet: str, *, encode: bool = True) -> List[str]:
    normalized = []
    for variant in (wallet, wallet.lower()):
        if variant and variant not in normalized:
            normalized.append(variant)
    if encode:
        return [urllib.parse.quote(variant, safe="0x") for variant in normalized]
    return normalized


def fetch_portfolio_snapshot(wallet: str) -> PortfolioSnapshot:
    """Fetch the live Polymarket portfolio for the given wallet address."""
    if not wallet:
        raise ValueError("wallet address is required")
    tried_urls: List[str] = []
    last_http_exc: Optional[urllib.error.HTTPError] = None
    auth_error: Optional[Exception] = None

    if _has_auth_credentials():
        try:
            payload, attempted = _fetch_authenticated_portfolio_payload(wallet)
            if attempted:
                tried_urls.extend(attempted)
            snapshot = _parse_snapshot(wallet, payload)
            if snapshot.cash_balance is None:
                snapshot.cash_balance = _fetch_cash_balance(wallet)
            return snapshot
        except Exception as exc:  # noqa: BLE001
            auth_error = exc

    for url in _portfolio_url_candidates(wallet):
        tried_urls.append(url)
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
                    raise RuntimeError(f"Portfolio API returned {response.status} {response.reason}")
                payload = json.load(response)
            snapshot = _parse_snapshot(wallet, payload)
            if snapshot.cash_balance is None:
                snapshot.cash_balance = _fetch_cash_balance(wallet)
            return snapshot
        except urllib.error.HTTPError as exc:
            last_http_exc = exc
            # Some deployments don't expose certain endpoints; try the next option on 404/405.
            if getattr(exc, "code", None) in (404, 405):
                continue
            raise RuntimeError(f"Portfolio API request failed ({exc.code}) for {url}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error contacting portfolio API ({url}): {exc}") from exc

    tried = ", ".join(tried_urls) if tried_urls else "none"
    message = (
        "Portfolio API request failed for all endpoints. "
        "You can override detection by setting POLYMARKET_PORTFOLIO_URL to a JSON endpoint or "
        "provide valid Polymarket API credentials. "
        f"Tried: {tried}"
    )
    if auth_error:
        message += f" (Authenticated request also failed: {auth_error})"
        raise RuntimeError(message) from auth_error
    if last_http_exc:
        raise RuntimeError(message) from last_http_exc
    raise RuntimeError(message)


def _parse_snapshot(wallet: str, payload: object) -> PortfolioSnapshot:
    snapshot = PortfolioSnapshot(wallet=wallet)
    raw_positions: List[dict] = []
    if isinstance(payload, dict):
        snapshot.realized_pnl = _safe_float(
            payload.get("realizedPnl")
            or payload.get("realizedPNL")
            or payload.get("realizedPnL")
            or payload.get("realized")
        )
        snapshot.unrealized_pnl = _safe_float(
            payload.get("unrealizedPnl")
            or payload.get("unrealizedPNL")
            or payload.get("unrealizedPnL")
            or payload.get("unrealized")
        )
        snapshot.cash_balance = _safe_float(
            payload.get("cashBalance") or payload.get("availableCash") or payload.get("usdBalance")
        )
        for key in ("positions", "data", "results"):
            maybe_positions = payload.get(key)
            if isinstance(maybe_positions, list):
                raw_positions = maybe_positions
                break
    elif isinstance(payload, list):
        raw_positions = payload

    for record in raw_positions:
        if not isinstance(record, dict):
            continue
        market_info = record.get("market")
        if not isinstance(market_info, dict):
            market_info = record.get("event")
        if not isinstance(market_info, dict):
            market_info = {}
        question = (
            record.get("question")
            or market_info.get("question")
            or market_info.get("title")
            or record.get("title")
            or record.get("marketQuestion")
            or record.get("slug")
            or "Unknown market"
        )
        slug = market_info.get("slug") or record.get("slug") or record.get("eventSlug")
        market_id = (
            record.get("marketId")
            or record.get("conditionId")
            or market_info.get("conditionId")
            or market_info.get("id")
            or "unknown"
        )
        outcome = (
            record.get("outcome")
            or record.get("side")
            or record.get("tokenName")
            or market_info.get("outcome")
            or "Unknown outcome"
        )
        shares = _safe_float(
            record.get("amount")
            or record.get("quantity")
            or record.get("qty")
            or record.get("netPosition")
            or record.get("balance")
            or record.get("size")
            or 0.0,
            default=0.0,
        )
        avg_price = _safe_float(
            record.get("avgPrice")
            or record.get("averagePrice")
            or record.get("entryPrice")
        )
        mark_price = _safe_float(
            record.get("markPrice")
            or record.get("currentPrice")
            or record.get("oraclePrice")
            or record.get("curPrice")
        )
        pnl = _safe_float(
            record.get("pnl")
            or record.get("unrealizedPnl")
            or record.get("realizedPnl")
            or record.get("pnlUsd")
            or record.get("cashPnl")
        )
        position = PortfolioPosition(
            market_id=str(market_id),
            question=str(question),
            outcome=str(outcome),
            shares=shares or 0.0,
            slug=str(slug) if slug else None,
            avg_price=avg_price,
            mark_price=mark_price,
            pnl=pnl,
        )
        snapshot.positions.append(position)
    return snapshot


__all__ = ["PortfolioPosition", "PortfolioSnapshot", "fetch_portfolio_snapshot"]


def _fetch_cash_balance(wallet: str) -> Optional[float]:
    """Fetch the custodial USDC balance via the balance-allowance endpoint."""
    if not wallet or not _has_auth_credentials():
        return None
    api_key = os.environ.get("POLYMARKET_API_KEY")
    api_secret = os.environ.get("POLYMARKET_SECRET")
    api_passphrase = os.environ.get("POLYMARKET_PASSPHRASE")
    if not (api_key and api_secret and api_passphrase):
        return None
    try:
        decimals = int(os.environ.get("POLYMARKET_CASH_DECIMALS", "6"))
    except ValueError:
        decimals = 6
    payload = _fetch_cash_payload_http(wallet, api_key, api_secret, api_passphrase)
    if payload is None:
        payload = _fetch_cash_payload_via_client(wallet)
    if not isinstance(payload, dict):
        return None
    raw_balance = payload.get("balance")
    try:
        scaled = int(raw_balance) / float(10**decimals)
    except (TypeError, ValueError):
        return None
    return scaled


def _fetch_cash_payload_http(
    wallet: str, api_key: str, api_secret: str, api_passphrase: str
) -> Optional[object]:
    api_root = os.environ.get("POLYMARKET_CASH_API_ROOT") or DEFAULT_API_ROOT
    api_root = api_root.rstrip("/")
    client = _AuthenticatedPolymarketClient(
        api_root=api_root,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        wallet_address=wallet,
    )
    params = {
        "asset_type": os.environ.get("POLYMARKET_BALANCE_ASSET_TYPE", "COLLATERAL"),
        "signature_type": 1,
    }
    token_id = os.environ.get("POLYMARKET_BALANCE_TOKEN_ID")
    if token_id is not None:
        params["token_id"] = token_id
    query = urllib.parse.urlencode(params, doseq=True)
    path = f"/balance-allowance?{query}"
    try:
        payload = client.get(path)
        return payload
    except Exception:
        if os.environ.get("POLYMARKET_DEBUG"):
            print("POLYMARKET_DEBUG: balance HTTP request failed.", flush=True)
    return None


def _fetch_cash_payload_via_client(wallet: str) -> Optional[object]:
    """Use py_clob_client if available to mirror the successful manual call."""
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        return None
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams, ApiCreds
    except ImportError:
        if os.environ.get("POLYMARKET_DEBUG"):
            print("POLYMARKET_DEBUG: py_clob_client not installed.", flush=True)
        return None

    api_root = os.environ.get("POLYMARKET_CASH_API_ROOT") or DEFAULT_API_ROOT
    signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1"))
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
    client = ClobClient(
        host=api_root,
        key=private_key,
        chain_id=chain_id,
        funder=wallet,
        signature_type=signature_type,
    )
    # Prefer explicit API credentials if supplied, otherwise derive them.
    api_key = os.environ.get("POLYMARKET_API_KEY")
    api_secret = os.environ.get("POLYMARKET_SECRET")
    api_passphrase = os.environ.get("POLYMARKET_PASSPHRASE")
    if api_key and api_secret and api_passphrase:
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
    else:
        creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    params = BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,
        token_id=os.environ.get("POLYMARKET_BALANCE_TOKEN_ID") or "",
        signature_type=signature_type,
    )
    try:
        return client.get_balance_allowance(params)
    except Exception:
        if os.environ.get("POLYMARKET_DEBUG"):
            print("POLYMARKET_DEBUG: py_clob_client balance fetch failed.", flush=True)
        return None


def _has_auth_credentials() -> bool:
    return all(
        os.environ.get(name)
        for name in ("POLYMARKET_API_KEY", "POLYMARKET_SECRET", "POLYMARKET_PASSPHRASE")
    )


def _fetch_authenticated_portfolio_payload(wallet: str) -> Tuple[object, List[str]]:
    api_key = os.environ.get("POLYMARKET_API_KEY")
    api_secret = os.environ.get("POLYMARKET_SECRET")
    api_passphrase = os.environ.get("POLYMARKET_PASSPHRASE")
    if not (api_key and api_secret and api_passphrase):
        raise RuntimeError("Polymarket API key, secret, and passphrase are required for authenticated portfolio access.")
    api_root = os.environ.get("POLYMARKET_API_ROOT", DEFAULT_API_ROOT).rstrip("/")
    client = _AuthenticatedPolymarketClient(
        api_root=api_root,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        wallet_address=wallet,
    )
    attempted: List[str] = []
    last_http_exc: Optional[urllib.error.HTTPError] = None
    for path in _auth_portfolio_paths(wallet):
        attempted.append(f"{api_root}{path}")
        try:
            payload = client.get(path)
            return payload, attempted
        except urllib.error.HTTPError as exc:
            last_http_exc = exc
            if getattr(exc, "code", None) in (401, 403, 404, 405):
                continue
            raise RuntimeError(f"Authenticated portfolio request failed ({exc.code}) for {path}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error contacting authenticated Polymarket API ({path}): {exc}") from exc
    message = (
        "Authenticated portfolio endpoints exhausted. "
        "Set POLYMARKET_AUTH_PORTFOLIO_PATH to the correct route if your account uses a custom deployment."
    )
    if last_http_exc:
        raise RuntimeError(message) from last_http_exc
    raise RuntimeError(message)


class _AuthenticatedPolymarketClient:
    """Minimal Polymarket API client that signs requests with HMAC headers."""

    def __init__(
        self,
        *,
        api_root: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        timeout: float = 10.0,
        wallet_address: Optional[str] = None,
    ) -> None:
        self._api_root = api_root.rstrip("/")
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._timeout = timeout
        self._wallet_address = wallet_address

    def get(self, path: str) -> object:
        return self._request("GET", path)

    def _request(self, method: str, path: str, body: object | None = None) -> object:
        url, request_path = self._build_url(path)
        body_text, body_bytes = self._serialize_body(body)
        timestamp = str(int(time.time()))
        signature = self._sign(timestamp, method.upper(), request_path, body_text)

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-KEY": self._api_key,
            "X-API-SIGNATURE": signature,
            "X-API-TIMESTAMP": timestamp,
            "X-API-PASSPHRASE": self._api_passphrase,
            "POLY_API_KEY": self._api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": self._api_passphrase,
        }
        if self._wallet_address:
            headers["POLY_ADDRESS"] = self._wallet_address

        request = urllib.request.Request(
            url,
            data=body_bytes,
            method=method.upper(),
            headers=headers,
        )
        if body_bytes is None:
            request.data = None  # type: ignore[assignment]
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url=url,
                    code=response.status,
                    msg=response.reason,
                    hdrs=response.headers,
                    fp=response,
                )
            return json.load(response)

    def _build_url(self, path: str) -> Tuple[str, str]:
        if not path:
            request_path = "/api/v1/portfolio"
        elif path.startswith("http://") or path.startswith("https://"):
            parsed = urllib.parse.urlparse(path)
            request_path = parsed.path or "/"
            if parsed.query:
                request_path += f"?{parsed.query}"
            url = path
            return url, request_path
        else:
            request_path = path if path.startswith("/") else f"/{path}"
        url = f"{self._api_root}{request_path}"
        return url, request_path

    def _serialize_body(self, body: object | None) -> Tuple[str, Optional[bytes]]:
        if body is None:
            return "", None
        if isinstance(body, bytes):
            return body.decode("utf-8"), body
        if isinstance(body, str):
            return body, body.encode("utf-8")
        text = json.dumps(body, separators=(",", ":"), sort_keys=True)
        return text, text.encode("utf-8")

    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        message = f"{timestamp}{method.upper()}{request_path}{body or ''}"
        secret_key = self._decode_secret()
        digest = hmac.new(secret_key, msg=message.encode("utf-8"), digestmod=hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _decode_secret(self) -> bytes:
        secret = self._api_secret.strip()
        base64_variants = [secret]
        if "-" in secret or "_" in secret:
            base64_variants.append(secret.replace("-", "+").replace("_", "/"))
        for candidate in base64_variants:
            padded = candidate + "=" * (-len(candidate) % 4)
            try:
                return base64.b64decode(padded, altchars=b"-_")
            except Exception:
                continue
        try:
            return bytes.fromhex(secret)
        except ValueError:
            pass
        raise RuntimeError("Invalid Polymarket API secret: must be base64-encoded or hex-encoded.")
