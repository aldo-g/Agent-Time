"""Error helpers for Manifold tools."""

from __future__ import annotations

import re
from typing import Optional

_SELL_CAP_PATTERN = re.compile(r"you can only sell up to ([0-9.eE+-]+) shares", re.IGNORECASE)
_NO_POSITION_TOKENS = (
    "no position",
    "position not found",
    "no shares",
    "no holding",
    "you don't have",
)


def _is_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "404" in message
        or "contract not found" in message
        or "market not found" in message
        or "unable to load manifold market" in message
    )


def _is_no_position_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(token in message for token in _NO_POSITION_TOKENS)


def _extract_sell_cap_shares(error: Exception) -> Optional[float]:
    """Parse Manifold's max-sellable-shares hint from an error message."""
    match = _SELL_CAP_PATTERN.search(str(error))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None
