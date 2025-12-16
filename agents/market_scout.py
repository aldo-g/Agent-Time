"""Simple analysis helper that inspects Manifold markets and explains findings."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class MarketHighlight:
    title: str
    details: str
    market_ids: Sequence[str] = field(default_factory=tuple)


@dataclass
class ScoutReport:
    fetched: int
    time_generated: datetime
    highlights: Sequence[MarketHighlight]
    next_steps: Sequence[str]
    focus_market_ids: Sequence[str] = field(default_factory=tuple)

    def as_text(self) -> str:
        lines = [
            f"Markets fetched: {self.fetched}",
            f"Generated at: {self.time_generated.isoformat()}",
            "",
            "Key observations:",
        ]
        for idx, highlight in enumerate(self.highlights, 1):
            lines.append(f"  {idx}. {highlight.title}")
            lines.append(f"     {highlight.details}")
        lines.append("")
        lines.append("Next actions:")
        for idx, step in enumerate(self.next_steps, 1):
            lines.append(f"  {idx}. {step}")
        return "\n".join(lines)


class MarketScout:
    """Heuristic-driven reasoning helper for the exploration phase."""

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now or datetime.now(timezone.utc)

    def analyze(self, markets: Sequence[Dict[str, Any]]) -> ScoutReport:
        highlights: List[MarketHighlight] = []
        soonest = self._soonest_closing(markets)
        if soonest:
            highlights.append(
                MarketHighlight(
                    title="Closing soon",
                    details=(
                        f"{soonest['question']} closes at {soonest['close_time'].isoformat()} "
                        f"with prob {soonest['prob']:.2f} and daily volume ${soonest['volume24h']:.0f}."
                    ),
                    market_ids=[soonest["id"]],
                )
            )
        liquid = self._highest_liquidity(markets)
        if liquid:
            highlights.append(
                MarketHighlight(
                    title="Most liquid markets",
                    details=", ".join(
                        f"{m['question']} (${m['liquidity']:.0f} pool)" for m in liquid
                    ),
                    market_ids=[m["id"] for m in liquid],
                )
            )
        activity = self._most_active(markets)
        if activity:
            highlights.append(
                MarketHighlight(
                    title="Highest 24h activity",
                    details=", ".join(
                        f"{m['question']} (${m['volume24h']:.0f})" for m in activity
                    ),
                    market_ids=[m["id"] for m in activity],
                )
            )
        next_steps = self._next_steps(markets, highlights)
        focus_market_ids = self._collect_market_ids(highlights)
        return ScoutReport(
            fetched=len(markets),
            time_generated=self._now,
            highlights=highlights,
            next_steps=next_steps,
            focus_market_ids=focus_market_ids,
        )

    def _soonest_closing(self, markets: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
        soonest: Dict[str, Any] | None = None
        for market in markets:
            market_id = market.get("id") or market.get("marketId")
            if not market_id:
                continue
            close_time = self._parse_timestamp(market.get("closeTime"))
            if close_time is None:
                continue
            if close_time < self._now:
                continue
            prob = float(market.get("probability", 0.0))
            volume24h = float(market.get("volume24Hours", 0.0) or market.get("volume24h", 0.0))
            candidate = {
                "id": str(market_id),
                "question": market.get("question", "<unknown>"),
                "close_time": close_time,
                "prob": prob,
                "volume24h": volume24h,
            }
            if soonest is None or candidate["close_time"] < soonest["close_time"]:
                soonest = candidate
        return soonest

    def _highest_liquidity(self, markets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = sorted(
            (
                {
                    "id": str(m.get("id") or m.get("marketId") or ""),
                    "question": m.get("question", "<unknown>"),
                    "liquidity": float(m.get("liquidity", 0.0)),
                }
                for m in markets
            ),
            key=lambda item: item["liquidity"],
            reverse=True,
        )
        return [m for m in ranked[:3] if m["liquidity"] > 0]

    def _most_active(self, markets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = sorted(
            (
                {
                    "id": str(m.get("id") or m.get("marketId") or ""),
                    "question": m.get("question", "<unknown>"),
                    "volume24h": float(m.get("volume24Hours", 0.0) or m.get("volume24h", 0.0)),
                }
                for m in markets
            ),
            key=lambda item: item["volume24h"],
            reverse=True,
        )
        return [m for m in ranked[:3] if m["volume24h"] > 0]

    def _next_steps(self, markets: Sequence[Dict[str, Any]], highlights: Sequence[MarketHighlight]) -> List[str]:
        steps = [
            "Compare current portfolio exposure vs. liquidity on target markets",
            "Decide whether to pursue momentum or mean-reversion on active markets",
        ]
        if not highlights:
            steps.insert(0, "Broaden market fetch parameters to find candidates (current set looks quiet)")
        if markets:
            steps.append("Fetch group metadata to cluster opportunities before executing trades")
        return steps

    @staticmethod
    def _collect_market_ids(highlights: Sequence[MarketHighlight]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for highlight in highlights:
            for market_id in highlight.market_ids:
                if not market_id or market_id in seen:
                    continue
                seen.add(market_id)
                ordered.append(market_id)
        return ordered

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                # Manifold timestamps are ms since epoch.
                return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
            if isinstance(value, str):
                # try parse ISO
                return datetime.fromisoformat(value)
        except Exception:
            return None
        return None
