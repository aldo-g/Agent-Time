"""Market inspection guards and normalization utilities."""

from __future__ import annotations

import json
import urllib.parse

from . import config

_INSPECTED_MARKETS: set[str] = set()


def _normalize_market_identifier(market_id: str) -> str:
    if not market_id:
        return market_id
    cleaned = market_id.strip().strip(".,;:!?)\"]'")
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("market_id", "marketId", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    cleaned = value.strip()
                    break
    lower = cleaned.lower()
    for prefix in ("market_id=", "market_id:", "id=", "id:"):
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip().strip(".,;:!?)\"]'")
            break
    if "manifold.markets" in cleaned:
        parsed = urllib.parse.urlparse(cleaned)
        path = parsed.path.strip("/")
        if path:
            return path.split("/")[-1].strip(".,;:!?)\"]'")
    if any(char.isspace() for char in cleaned):
        cleaned = cleaned.split()[0]
    return cleaned


def reset_inspected_markets() -> None:
    """Reset the per-run market inspection tracker."""
    _INSPECTED_MARKETS.clear()


def _enforce_market_limit(market_id: str) -> tuple[bool, str, str | None]:
    """Allow at most MARKET_INSPECTION_LIMIT distinct markets per run for deep tool calls."""
    normalized = _normalize_market_identifier(market_id)
    if normalized in _INSPECTED_MARKETS:
        return True, normalized, None
    if len(_INSPECTED_MARKETS) >= config.MARKET_INSPECTION_LIMIT:
        inspected_preview = ", ".join(sorted(_INSPECTED_MARKETS)) or "none"
        return (
            False,
            normalized,
            f"Market limit reached ({config.MARKET_INSPECTION_LIMIT} distinct markets already inspected: {inspected_preview}). "
            "Reuse one of those or adjust AGENT_MARKET_INSPECTION_LIMIT.",
        )
    _INSPECTED_MARKETS.add(normalized)
    return True, normalized, None
