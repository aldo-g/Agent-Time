"""Manifold Markets API client (read-only subset)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

API_ROOT = os.environ.get("MANIFOLD_API_ROOT", "https://api.manifold.markets/v0")


class ManifoldAPIError(RuntimeError):
    """Raised when the Manifold API returns an unexpected response."""


@dataclass(frozen=True)
class ManifoldMarketProb:
    market_id: str
    probability: float


class ManifoldClient:
    def __init__(
        self,
        api_root: str = API_ROOT,
        session: Optional[requests.Session] = None,
        timeout: Optional[float] = 10.0,
        api_key: Optional[str] = None,
    ) -> None:
        self._api_root = api_root.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._api_key = api_key or os.environ.get("MANIFOLD_API_KEY")
        if self._api_key:
            self._session.headers.update({"Authorization": f"Key {self._api_key}"})

    def list_markets(self, **params: Any) -> List[Dict[str, Any]]:
        """GET /v0/markets"""
        return self._get("/markets", params=params)

    def get_market(self, market_id: str) -> Dict[str, Any]:
        """GET /v0/market/:id"""
        return self._get(f"/market/{market_id}")

    def get_market_prob(self, market_id: str) -> ManifoldMarketProb:
        """GET /v0/market/:id/prob"""
        data = self._get(f"/market/{market_id}/prob")
        if not isinstance(data, dict) or "probability" not in data:
            raise ManifoldAPIError("/prob endpoint returned unexpected payload")
        return ManifoldMarketProb(market_id=market_id, probability=float(data["probability"]))

    def list_market_probs(self, market_ids: Iterable[str]) -> List[ManifoldMarketProb]:
        joined_ids = ",".join(market_ids)
        payload = self._get("/market-probs", params={"ids": joined_ids})
        if not isinstance(payload, dict):
            raise ManifoldAPIError("/market-probs returned unexpected payload")
        return [
            ManifoldMarketProb(market_id=mid, probability=float(prob))
            for mid, prob in payload.items()
        ]

    def list_groups(self, **params: Any) -> List[Dict[str, Any]]:
        """GET /v0/groups"""
        return self._get("/groups", params=params)

    def get_group(self, slug: str) -> Dict[str, Any]:
        """GET /v0/group/:slug"""
        return self._get(f"/group/{slug}")

    def get_me(self) -> Dict[str, Any]:
        """GET /v0/me (requires MANIFOLD_API_KEY)"""
        return self._get("/me")

    def get_portfolio(self) -> Dict[str, Any]:
        """GET /v0/me (requires MANIFOLD_API_KEY)."""
        data = self.get_me()
        if not isinstance(data, dict):
            raise ManifoldAPIError("/me returned unexpected payload")
        return data

    # CamelCase wrappers to match earlier plan terminology.
    def listMarkets(self, **params: Any) -> List[Dict[str, Any]]:
        return self.list_markets(**params)

    def getMarket(self, market_id: str) -> Dict[str, Any]:
        return self.get_market(market_id)

    def getMarketProb(self, market_id: str) -> ManifoldMarketProb:
        return self.get_market_prob(market_id)

    def listGroups(self, **params: Any) -> List[Dict[str, Any]]:
        return self.list_groups(**params)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self._api_root}{path}"
        query = dict(params or {})
        self._debug_request(url, query)
        response = self._session.get(
            url,
            params=query or None,
            timeout=self._timeout,
        )
        if not response.ok:
            raise ManifoldAPIError(f"GET {url} failed: {response.status_code} {response.text}")
        return response.json()

    def _debug_request(self, url: str, params: Optional[Dict[str, Any]]) -> None:
        if not os.environ.get("MANIFOLD_DEBUG"):
            return
        auth_header = self._session.headers.get("Authorization")
        masked = self._mask_auth(auth_header)
        payload = ""
        if params:
            safe_params = {
                key: (self._mask_auth(str(value)) if key.lower() == "apikey" else value)
                for key, value in params.items()
            }
            payload = f" params={safe_params}"
        print(f"[manifold] GET {url}{payload} auth={masked}")

    @staticmethod
    def _mask_auth(value: Optional[str]) -> str:
        if not value:
            return "<none>"
        if len(value) <= 10:
            return value
        return value[:6] + "…" + value[-4:]
