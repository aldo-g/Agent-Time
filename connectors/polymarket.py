"""Polymarket CLOB API client (read-only subset)."""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_PRIMARY_ROOT = "https://clob.polymarket.com"
FALLBACK_ROOT = "https://gamma-api.polymarket.com"


class PolymarketAPIError(RuntimeError):
    """Raised when the Polymarket API returns an unexpected response."""


class PolymarketClient:
    """Thin wrapper over the Polymarket CLOB API with legacy field compatibility."""

    supports_portfolio: bool = False

    def __init__(
        self,
        api_root: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[float] = 10.0,
        wallet_address: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
    ) -> None:
        env_root = os.environ.get("POLYMARKET_API_ROOT")
        if api_root:
            primary_root = api_root
            fallback_root = None
        elif env_root:
            primary_root = env_root
            fallback_root = None
        else:
            primary_root = DEFAULT_PRIMARY_ROOT
            fallback_root = FALLBACK_ROOT
        self._active_root = primary_root.rstrip("/")
        self._fallback_root = fallback_root.rstrip("/") if fallback_root else None
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._timeout = timeout
        self._wallet_address = wallet_address or os.environ.get("POLYMARKET_WALLET_ADDRESS")
        self._api_key = api_key or os.environ.get("POLYMARKET_API_KEY")
        self._api_secret = api_secret or os.environ.get("POLYMARKET_SECRET")
        self._api_passphrase = api_passphrase or os.environ.get("POLYMARKET_PASSPHRASE")
        self._last_error: str | None = None
        # Portfolio endpoints require signing; disable until implemented.
        self.supports_portfolio = False

    def list_markets(self, limit: int = 50, sort: str | None = None, **params: Any) -> List[Dict[str, Any]]:
        """Return recent markets with normalized liquidity/probability fields."""
        query: Dict[str, Any] = {"limit": limit}
        offset = params.pop("offset", None)
        if offset is not None:
            query["offset"] = offset
        active = params.pop("active", True)
        if active is not None:
            query["active"] = str(bool(active)).lower()
        query.update(params)
        payload = self._get("/markets", params=query)
        items = self._extract_list(payload)
        normalized = [self._normalize_market(item) for item in items]
        open_markets = [item for item in normalized if self._is_open_market(item)]
        closing_soon = [item for item in open_markets if self._closes_within(item, hours=24)]
        selected = closing_soon or open_markets or normalized
        return self._sort_markets(selected, sort)

    def get_market(self, market_id: str) -> Dict[str, Any]:
        try:
            payload = self._get(f"/markets/{market_id}")
        except PolymarketAPIError:
            cached = self._lookup_market_from_feed(market_id)
            if cached is None:
                raise
            return cached
        data = self._extract_dict(payload)
        normalized = self._normalize_market(data)
        normalized["raw"] = data
        return normalized

    def get_market_bets(
        self,
        market_id: str,
        *,
        limit: int = 100,
        before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent fills for a given market."""
        params: Dict[str, Any] = {"market": market_id, "limit": limit}
        if before:
            params["before"] = before
        payload = self._get("/fills", params=params)
        fills = self._extract_list(payload)
        return [self._normalize_fill(fill) for fill in fills]

    def get_portfolio(self) -> Dict[str, Any]:
        raise PolymarketAPIError("Portfolio endpoint is not yet available for Polymarket.")

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _normalize_market(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = self._flatten_market_payload(payload)
        market_id = self._extract_market_id(source)
        if not market_id:
            keys = ", ".join(sorted(source.keys()))
            self._record_error(f"Market missing identifier. keys=[{keys}]")
            raise PolymarketAPIError("Market missing identifier.")
        normalized = dict(source)
        normalized["id"] = str(market_id)
        normalized["marketId"] = normalized["id"]
        normalized["question"] = str(
            source.get("question") or source.get("title") or source.get("name") or "Untitled market"
        )
        probability = self._derive_probability(source)
        normalized["probability"] = probability
        volume24h = self._extract_float(source, ("volume24h", "volume24Hr", "volume24Hours", "volume24Hour", "volume24"))
        normalized["volume24h"] = volume24h
        normalized["volume24Hours"] = volume24h
        liquidity = self._extract_float(source, ("liquidity", "openInterest", "liquidity24h", "pool"))
        normalized["liquidity"] = liquidity
        close_iso = self._format_timestamp(
            source,
            keys=(
                "closeTime",
                "closeDate",
                "endDate",
                "closesAt",
                "expiry",
                "expiration",
                "resolveTime",
            ),
        )
        if close_iso:
            normalized["closeTime"] = close_iso
            normalized["close_time"] = close_iso
        updated_iso = self._format_timestamp(source, keys=("lastTradeTime", "updatedTime", "updatedAt"))
        if updated_iso:
            normalized["lastBetTime"] = updated_iso
            normalized["updatedTime"] = updated_iso
        normalized["url"] = source.get("url") or self._build_market_url(source)
        normalized["creatorUsername"] = (
            source.get("creatorUsername") or source.get("creator") or source.get("author") or ""
        )
        normalized.setdefault("mechanism", source.get("mechanism") or "clob")
        return normalized

    def _extract_market_id(self, source: Dict[str, Any]) -> str | None:
        def first_truthy(values: List[Any]) -> Any:
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value
                if value not in (None, "", 0, False):
                    return value
            return None

        candidates: List[Any] = [
            source.get("id"),
            source.get("marketId"),
            source.get("conditionId"),
            source.get("condition_id"),
            source.get("question_id"),
            source.get("marketHash"),
            source.get("_id"),
            source.get("market_slug"),
            source.get("slug"),
        ]
        fpmm = source.get("fpmm")
        if isinstance(fpmm, dict):
            candidates.extend(
                [
                    fpmm.get("conditionId"),
                    fpmm.get("condition_id"),
                    fpmm.get("marketHash"),
                    fpmm.get("_id"),
                ]
            )
        tokens = source.get("tokens")
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                candidates.extend(
                    [
                        token.get("market_hash"),
                        token.get("marketHash"),
                        token.get("condition_id"),
                        token.get("conditionId"),
                        token.get("token_id"),
                        token.get("tokenId"),
                    ]
                )
        candidate = first_truthy(candidates)
        if candidate is None:
            return None
        return str(candidate)

    def _is_open_market(self, market: Dict[str, Any]) -> bool:
        if market.get("closed"):
            return False
        if not market.get("active", True):
            return False
        return True

    def _closes_within(self, market: Dict[str, Any], *, hours: float) -> bool:
        close_value = (
            market.get("close_time")
            or market.get("closeTime")
            or market.get("endDate")
            or market.get("end_date_iso")
        )
        close_dt = self._parse_datetime(close_value)
        if not close_dt:
            return False
        now = datetime.now(timezone.utc)
        if close_dt < now:
            return False
        return close_dt <= now + timedelta(hours=hours)

    def _flatten_market_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = dict(payload)
        for key in ("market", "contract", "data"):
            inner = source.get(key)
            if isinstance(inner, dict) and inner:
                merged = dict(inner)
                for outer_key, outer_value in source.items():
                    if outer_key in (key,):
                        continue
                    merged.setdefault(outer_key, outer_value)
                source = merged
        return source

    def _normalize_fill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)
        created = payload.get("createdTime") or payload.get("timestamp") or payload.get("createdAt")
        created_ms = self._to_millis(created)
        if created_ms is not None:
            normalized["createdTime"] = created_ms
        price = payload.get("probAfter") or payload.get("price") or payload.get("avgPrice") or payload.get("executionPrice")
        if price is not None:
            try:
                normalized["probAfter"] = float(price)
            except Exception:
                pass
        return normalized

    def _extract_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "markets", "results", "items", "fills"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise PolymarketAPIError("Expected list payload from Polymarket API.")

    def _extract_dict(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            if "id" in payload or "question" in payload:
                return payload
            for key in ("data", "market", "result"):
                inner = payload.get(key)
                if isinstance(inner, dict):
                    return inner
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    return entry
        raise PolymarketAPIError("Expected dict payload from Polymarket API.")

    def _sort_markets(self, markets: Sequence[Dict[str, Any]], sort: str | None) -> List[Dict[str, Any]]:
        if not sort:
            return list(markets)
        sort_key = sort.lower()
        if sort_key in {"last-bet-time", "updated-time", "last-comment-time"}:
            return sorted(
                markets,
                key=lambda m: self._sort_timestamp(m.get("lastBetTime") or m.get("updatedTime") or m.get("closeTime")),
                reverse=True,
            )
        if sort_key in {"volume24h", "volume", "24hvolume"}:
            return sorted(markets, key=lambda m: float(m.get("volume24h", 0.0)), reverse=True)
        if sort_key in {"liquidity"}:
            return sorted(markets, key=lambda m: float(m.get("liquidity", 0.0)), reverse=True)
        return list(markets)

    def _sort_timestamp(self, value: Any) -> float:
        dt = self._parse_datetime(value)
        if dt:
            return dt.timestamp()
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    def _format_timestamp(self, payload: Dict[str, Any], keys: Sequence[str]) -> str | None:
        for key in keys:
            if key in payload:
                dt = self._parse_datetime(payload[key])
                if dt:
                    return dt.isoformat()
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 1e12:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text)
            except Exception:
                return None
        return None

    def _to_millis(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value > 1e12:
                return int(value)
            return int(float(value) * 1000)
        dt = self._parse_datetime(value)
        if dt:
            return int(dt.timestamp() * 1000)
        return None

    def _extract_float(self, payload: Dict[str, Any], keys: Sequence[str]) -> float:
        for key in keys:
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except Exception:
                    continue
        return 0.0

    def _derive_probability(self, payload: Dict[str, Any]) -> float:
        for key in ("probability", "pYes", "yesPrice", "mid", "price", "lastPrice", "bestBid", "impliedProbability"):
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except Exception:
                    continue
        outcomes = payload.get("outcomePrices") or payload.get("outcomes")
        if isinstance(outcomes, dict):
            for candidate in ("YES", "Yes", "Y", "1", "UP", "TRUE"):
                if candidate in outcomes:
                    try:
                        return float(outcomes[candidate])
                    except Exception:
                        continue
        if isinstance(outcomes, list):
            for outcome in outcomes:
                name = str(outcome.get("name") or outcome.get("outcome") or "").lower()
                if name in {"yes", "y", "1", "up", "true"}:
                    for key in ("price", "probability", "p"):
                        if key in outcome and outcome[key] is not None:
                            try:
                                return float(outcome[key])
                            except Exception:
                                continue
        return 0.0

    def _build_market_url(self, payload: Dict[str, Any]) -> str:
        slug = payload.get("slug") or payload.get("urlSlug")
        if slug:
            return f"https://polymarket.com/market/{slug}"
        return "https://polymarket.com/"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        query = dict(params or {})
        last_error: Exception | None = None
        self._last_error = None
        for root in self._candidate_roots():
            url = f"{root}{path}"
            self._debug_request(url, query)
            try:
                response = self._session.get(
                    url,
                    params=query or None,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                self._record_error(f"Request failure for {url}: {exc}")
                continue
            if not response.ok:
                message = f"GET {url} failed: {response.status_code} {response.text}"
                last_error = PolymarketAPIError(message)
                self._record_error(message)
                continue
            try:
                data = response.json()
            except ValueError as exc:
                message = f"GET {url} returned invalid JSON: {exc}"
                last_error = PolymarketAPIError(message)
                body_excerpt = response.text[:200] if hasattr(response, "text") else ""
                self._record_error(f"{message} body={body_excerpt!r}")
                continue
            if root != self._active_root:
                self._active_root = root
                self._fallback_root = None
            self._last_error = None
            return data
        raise PolymarketAPIError(str(last_error) if last_error else "Unknown Polymarket API error.")

    def _debug_request(self, url: str, params: Optional[Dict[str, Any]]) -> None:
        if not os.environ.get("POLYMARKET_DEBUG"):
            return
        payload = f" params={params}" if params else ""
        print(f"[polymarket] GET {url}{payload}")

    def _record_error(self, message: str) -> None:
        self._last_error = message
        if os.environ.get("POLYMARKET_DEBUG"):
            print(f"[polymarket] ERROR {message}")

    def _candidate_roots(self) -> List[str]:
        roots = [self._active_root]
        if self._fallback_root and self._fallback_root not in roots:
            roots.append(self._fallback_root)
        return roots

    def _lookup_market_from_feed(self, market_id: str) -> Dict[str, Any] | None:
        try:
            markets = self.list_markets(limit=250)
        except PolymarketAPIError:
            return None
        target = str(market_id)
        for market in markets:
            if str(market.get("id") or market.get("marketId")) == target:
                return market
        return None
