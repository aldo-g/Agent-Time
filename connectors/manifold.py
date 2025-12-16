"""Manifold Markets API client (read-only subset)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

API_ROOT = "https://manifold.markets/api/v0"


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
    ) -> None:
        self._api_root = api_root.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout

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
        response = self._session.get(url, params=params, timeout=self._timeout)
        if not response.ok:
            raise ManifoldAPIError(f"GET {url} failed: {response.status_code} {response.text}")
        return response.json()
