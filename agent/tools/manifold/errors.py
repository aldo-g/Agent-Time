"""Error helpers for Manifold tools."""

from __future__ import annotations

import re
from typing import Optional

_SELL_CAP_PATTERN = re.compile(r"you can only sell up to ([0-9.eE+-]+) shares", re.IGNORECASE)


def _is_not_found_error(error: Exception) -> bool:
    message = str(error).lower()
    return "404" in message or "not found" in message or "contract not found" in message


def _extract_sell_cap_shares(error: Exception) -> Optional[float]:
    """Parse Manifold's max-sellable-shares hint from an error message."""
    match = _SELL_CAP_PATTERN.search(str(error))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None
