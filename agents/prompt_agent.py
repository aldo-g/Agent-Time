"""Prompt-driven agent that fetches markets and builds a plan of action."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

from agents.market_scout import MarketScout
from connectors.manifold import ManifoldClient, ManifoldAPIError
from core.agent_interface import PortfolioSnapshot


@dataclass
class AgentThought:
    heading: str
    detail: str


@dataclass
class AgentPlan:
    timestamp: datetime
    thoughts: Sequence[AgentThought]
    next_actions: Sequence[str]
    market_details: Sequence["MarketDetail"] = field(default_factory=tuple)
    portfolio: PortfolioSnapshot | None = None
    portfolio_notes: Sequence[str] = field(default_factory=tuple)

    def as_text(self) -> str:
        lines = [f"Agent run at: {self.timestamp.isoformat()}", "", "Thoughts:"]
        for idx, thought in enumerate(self.thoughts, 1):
            lines.append(f"  {idx}. {thought.heading}")
            lines.append(f"     {thought.detail}")
        lines.append("")
        if self.market_details:
            lines.append("Market details:")
            for idx, detail in enumerate(self.market_details, 1):
                close_desc = detail.close_time.isoformat() if detail.close_time else "unknown close"
                lines.append(f"  {idx}. {detail.question}")
                lines.append(
                    f"     Prob {detail.probability:.2f} | 24h vol ${detail.volume24h:.0f} | Close {close_desc}"
                )
                lines.append(f"     URL: {detail.url}")
            lines.append("")
        if self.portfolio is not None:
            lines.append("Portfolio snapshot:")
            cash = self.portfolio.cash_balance
            invested = self.portfolio.total_exposure
            lines.append(f"  Cash balance: ${cash:,.0f}")
            lines.append(f"  Active investment value: ${invested:,.0f}")
            lines.append(f"  Total buying power: ${(cash + invested):,.0f}")
            lines.append("")
        if self.portfolio_notes:
            lines.append("Portfolio insights:")
            for idx, note in enumerate(self.portfolio_notes, 1):
                lines.append(f"  {idx}. {note}")
            lines.append("")
        lines.append("Next actions:")
        for idx, action in enumerate(self.next_actions, 1):
            lines.append(f"  {idx}. {action}")
        return "\n".join(lines)


@dataclass
class MarketDetail:
    market_id: str
    question: str
    url: str
    probability: float
    volume24h: float
    liquidity: float
    close_time: datetime | None
    mechanism: str | None = None
    creator_username: str | None = None


class PromptAgentError(RuntimeError):
    pass


ALLOWED_SORTS = {"created-time", "updated-time", "last-bet-time", "last-comment-time"}


class PromptAgent:
    def __init__(self, client: ManifoldClient | None = None) -> None:
        self._client = client or ManifoldClient()

    def run(self, limit: int = 40, sort: str = "last-bet-time") -> AgentPlan:
        markets = self._fetch_markets(limit=limit, sort=sort)
        scout = MarketScout()
        report = scout.analyze(markets)
        thoughts = [
            AgentThought(heading=h.title, detail=h.details) for h in report.highlights
        ]
        if not thoughts:
            thoughts.append(
                AgentThought(
                    heading="No obvious opportunities found",
                    detail="Fetched markets but none passed the heuristic filters; broaden search.",
                )
            )
        market_details = self._fetch_market_details(report.focus_market_ids)
        portfolio, portfolio_error = self._fetch_portfolio_snapshot()
        portfolio_notes = self._assess_portfolio_liquidity(portfolio, market_details)
        if portfolio_error:
            portfolio_notes = tuple(portfolio_notes) + (portfolio_error,)
        next_actions = self._next_action_list(
            report.next_steps, completed_portfolio_check=(portfolio is not None)
        )
        return AgentPlan(
            timestamp=datetime.now(timezone.utc),
            thoughts=thoughts,
            next_actions=next_actions,
            market_details=market_details,
            portfolio=portfolio,
            portfolio_notes=portfolio_notes,
        )

    def _fetch_markets(self, limit: int, sort: str) -> Sequence[Dict[str, Any]]:
        normalized_sort = self._normalize_sort(sort)
        try:
            return self._client.list_markets(limit=limit, sort=normalized_sort)
        except ManifoldAPIError as exc:
            raise PromptAgentError(str(exc)) from exc

    def _normalize_sort(self, sort: str) -> str:
        sort_lower = sort.lower()
        mapping = {
            "24hourvolume": "last-bet-time",
            "24hvolume": "last-bet-time",
            "volume24hours": "last-bet-time",
            "volume": "last-bet-time",
            "liquidity": "last-bet-time",
        }
        if sort_lower in mapping:
            return mapping[sort_lower]
        if sort in ALLOWED_SORTS:
            return sort
        if sort_lower in ALLOWED_SORTS:
            # user typed e.g. LAST-BET-TIME
            return sort_lower
        return "last-bet-time"

    def _fetch_portfolio_snapshot(self) -> tuple[PortfolioSnapshot | None, str | None]:
        try:
            payload = self._client.get_portfolio()
        except ManifoldAPIError as exc:
            return None, f"Portfolio fetch failed: {exc}"
        cash_balance = self._extract_cash_balance(payload)
        investment_value = self._extract_investment_value(payload)
        snapshot = PortfolioSnapshot(
            cash_balance=cash_balance,
            total_exposure=investment_value,
            positions=tuple(),
        )
        return snapshot, None

    @staticmethod
    def _extract_cash_balance(payload: Dict[str, Any]) -> float:
        for key in ("balance", "cashBalance", "cash", "availableBalance"):
            if key in payload:
                try:
                    return float(payload[key] or 0.0)
                except Exception:
                    continue
        return 0.0

    @staticmethod
    def _extract_investment_value(payload: Dict[str, Any]) -> float:
        for key in ("investmentValue", "investedValue", "portfolioValue", "totalSharesValue"):
            if key in payload:
                try:
                    return float(payload[key] or 0.0)
                except Exception:
                    continue
        return 0.0

    def _fetch_market_details(self, market_ids: Sequence[str], limit: int = 5) -> Sequence[MarketDetail]:
        ordered: List[str] = []
        seen = set()
        for market_id in market_ids:
            if not market_id or market_id in seen:
                continue
            seen.add(market_id)
            ordered.append(market_id)
        details: List[MarketDetail] = []
        for market_id in ordered[:limit]:
            try:
                payload = self._client.get_market(market_id)
            except ManifoldAPIError:
                continue
            detail = self._convert_market_detail(payload)
            if detail is not None:
                details.append(detail)
        return details

    def _convert_market_detail(self, payload: Dict[str, Any]) -> MarketDetail | None:
        market_id = payload.get("id") or payload.get("marketId")
        question = payload.get("question")
        if not market_id or not question:
            return None
        close_time = self._parse_timestamp(payload.get("closeTime"))
        url = payload.get("url") or self._fallback_url(payload)
        probability = float(payload.get("probability", 0.0) or 0.0)
        volume24h = float(payload.get("volume24Hours", 0.0) or payload.get("volume24h", 0.0) or 0.0)
        liquidity = float(payload.get("liquidity", 0.0) or 0.0)
        mechanism = payload.get("mechanism")
        creator_username = payload.get("creatorUsername")
        return MarketDetail(
            market_id=str(market_id),
            question=str(question),
            url=url,
            probability=probability,
            volume24h=volume24h,
            liquidity=liquidity,
            close_time=close_time,
            mechanism=mechanism,
            creator_username=creator_username,
        )

    def _assess_portfolio_liquidity(
        self, portfolio: PortfolioSnapshot | None, market_details: Sequence[MarketDetail]
    ) -> Sequence[str]:
        if portfolio is None:
            return ()
        notes: List[str] = []
        cash = max(0.0, portfolio.cash_balance)
        invested = max(0.0, portfolio.total_exposure)
        total_power = cash + invested
        notes.append(
            f"Cash ${cash:,.0f} vs invested ${invested:,.0f}; total buying power ${total_power:,.0f}."
        )
        if not market_details:
            notes.append("No highlighted markets available for liquidity comparison.")
            return notes
        for detail in market_details:
            liquidity = max(0.0, detail.liquidity)
            if liquidity <= 0:
                continue
            safe_cash_slice = cash * 0.05
            safe_liquidity_slice = liquidity * 0.10
            suggested = min(safe_cash_slice if safe_cash_slice > 0 else cash, safe_liquidity_slice)
            ratio = (invested / liquidity) if liquidity else 0.0
            notes.append(
                f"{detail.question} pool ${liquidity:,.0f}; exposure/liquidity ratio {ratio:.2f}."
                f" Suggest staking around ${suggested:,.0f} (<=5% cash & <=10% pool)."
            )
        return notes

    @staticmethod
    def _next_action_list(steps: Sequence[str], completed_portfolio_check: bool) -> List[str]:
        if not completed_portfolio_check:
            return list(steps)
        filtered: List[str] = []
        for step in steps:
            if "Compare current portfolio exposure" in step:
                continue
            filtered.append(step)
        return filtered or list(steps)

    @staticmethod
    def _fallback_url(payload: Dict[str, Any]) -> str:
        slug = payload.get("slug") or ""
        creator = payload.get("creatorUsername") or ""
        if slug and creator:
            return f"https://manifold.markets/{creator}/{slug}"
        return "https://manifold.markets/"

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
            if isinstance(value, str):
                return datetime.fromisoformat(value)
        except Exception:
            return None
        return None
