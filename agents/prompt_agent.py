"""Prompt-driven agent that fetches Polymarket data and builds a plan of action."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from math import inf
from typing import Any, Dict, List, Sequence

from agents.market_scout import MarketScout
from connectors.polymarket import PolymarketClient, PolymarketAPIError
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
    research_briefs: Sequence["ResearchBrief"] = field(default_factory=tuple)
    bet_recommendations: Sequence["BetRecommendation"] = field(default_factory=tuple)
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
        if self.research_briefs:
            lines.append("Research briefs:")
            for idx, brief in enumerate(self.research_briefs, 1):
                lines.append(f"  {idx}. {brief.market_title}")
                lines.append(f"     Initial view: {brief.initial_view}")
                if brief.crowd_trend:
                    lines.append(f"     Crowd trend: {brief.crowd_trend}")
                if brief.research_queries:
                    lines.append("     Research to run:")
                    for query in brief.research_queries:
                        lines.append(f"       - {query}")
                if brief.research_summary:
                    lines.append(f"     Findings: {brief.research_summary}")
                lines.append(f"     Next step: {brief.recommended_direction}")
                if brief.recommended_amount:
                    lines.append(f"     Suggested stake: ${brief.recommended_amount:,.0f}")
                elif brief.max_stake is not None:
                    lines.append(f"     Stake cap (post research): ${brief.max_stake:,.0f}")
                lines.append(f"     Confidence: {brief.recommended_confidence:.0%}")
            lines.append("")
        if self.bet_recommendations:
            lines.append("Bet recommendations:")
            for idx, rec in enumerate(self.bet_recommendations, 1):
                lines.append(
                    f"  {idx}. {rec.market_title}: {rec.action} for ${rec.amount:,.0f} (confidence {rec.confidence:.0%})"
                )
                lines.append(f"     Rationale: {rec.rationale}")
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


@dataclass
class ResearchBrief:
    market_id: str
    market_title: str
    market_probability: float
    initial_view: str
    research_queries: Sequence[str]
    proposed_trade: str
    confidence: float
    stake_fraction: float
    liquidity: float
    max_stake: float | None = None
    crowd_change: float = 0.0
    crowd_volatility: float = 0.0
    crowd_samples: int = 0
    crowd_trend: str = ""
    research_summary: str = ""
    recommended_direction: str = "Pending research"
    recommended_confidence: float = 0.0
    recommended_amount: float = 0.0


@dataclass
class BetRecommendation:
    market_id: str
    market_title: str
    action: str
    amount: float
    rationale: str
    confidence: float


class PromptAgentError(RuntimeError):
    pass


ALLOWED_SORTS = {"last-bet-time", "updated-time", "volume24h", "liquidity"}


class PromptAgent:
    def __init__(self, client: PolymarketClient | None = None) -> None:
        self._client = client or PolymarketClient()

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
        research_briefs = self._build_research_briefs(market_details)
        self._attach_market_history(research_briefs)
        portfolio, portfolio_error = self._fetch_portfolio_snapshot()
        research_briefs = self._apply_stake_limits(research_briefs, portfolio)
        bet_recommendations = self._run_research(research_briefs)
        portfolio_notes = self._assess_portfolio_liquidity(portfolio, market_details)
        if portfolio_error:
            portfolio_notes = tuple(portfolio_notes) + (portfolio_error,)
        next_actions = self._next_action_list(report.next_steps, research_briefs, bet_recommendations)
        return AgentPlan(
            timestamp=datetime.now(timezone.utc),
            thoughts=thoughts,
            next_actions=next_actions,
            market_details=market_details,
            research_briefs=research_briefs,
            bet_recommendations=bet_recommendations,
            portfolio=portfolio,
            portfolio_notes=portfolio_notes,
        )

    def _fetch_markets(self, limit: int, sort: str) -> Sequence[Dict[str, Any]]:
        normalized_sort = self._normalize_sort(sort)
        try:
            return self._client.list_markets(limit=limit, sort=normalized_sort)
        except PolymarketAPIError as exc:
            detail = self._client.last_error
            message = str(exc)
            if detail and detail not in message:
                message = f"{message} | detail: {detail}"
            raise PromptAgentError(message) from exc

    def _normalize_sort(self, sort: str) -> str:
        sort_lower = sort.lower()
        mapping = {
            "24hourvolume": "volume24h",
            "24hvolume": "volume24h",
            "volume24hours": "volume24h",
            "volume": "volume24h",
            "liquidity": "liquidity",
            "created-time": "last-bet-time",
            "last-comment-time": "last-bet-time",
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
        if not getattr(self._client, "supports_portfolio", False):
            return None, "Portfolio fetch unavailable for this connector."
        try:
            payload = self._client.get_portfolio()
        except PolymarketAPIError as exc:
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
            except PolymarketAPIError:
                continue
            detail = self._convert_market_detail(payload)
            if detail is not None:
                details.append(detail)
        return details

    def _build_research_briefs(self, markets: Sequence[MarketDetail]) -> Sequence[ResearchBrief]:
        briefs: List[ResearchBrief] = []
        now = datetime.now(timezone.utc)
        for detail in markets:
            closing_hours = self._hours_until_close(detail.close_time, now)
            probability = detail.probability
            liquidity = detail.liquidity
            volume = detail.volume24h
            initial_view = self._initial_take(detail.question, probability, closing_hours)
            research_queries = self._research_queries(detail.question)
            proposed_trade, confidence, stake_fraction = self._proposed_trade(probability, liquidity, volume)
            briefs.append(
                ResearchBrief(
                    market_id=detail.market_id,
                    market_title=detail.question,
                    market_probability=probability,
                    initial_view=initial_view,
                    research_queries=research_queries,
                    proposed_trade=proposed_trade,
                    confidence=confidence,
                    stake_fraction=stake_fraction,
                    liquidity=liquidity,
                )
            )
        return briefs

    @staticmethod
    def _hours_until_close(close_time: datetime | None, now: datetime) -> float:
        if close_time is None:
            return inf
        delta = close_time - now
        return max(delta.total_seconds() / 3600.0, 0.0)

    def _initial_take(self, question: str, probability: float, closing_hours: float) -> str:
        window = f"closes in {closing_hours:.1f}h" if closing_hours != inf else "no close date found"
        lean = "balanced" if 0.35 <= probability <= 0.65 else ("likely" if probability > 0.65 else "unlikely")
        return f"{window}; market currently prices this as {lean} ({probability:.2f})."

    def _research_queries(self, question: str) -> List[str]:
        base = question.split("?")[0]
        queries = [
            f'"{base}" latest news',
            f'"{base}" site:.gov data 2025',
        ]
        if "election" in base.lower() or "Trump" in base or "Biden" in base:
            queries.append("polling data 2025 election forecasting")
        if "war" in base.lower():
            queries.append("geopolitical risk report 2025")
        if any(word in base.lower() for word in ("stock", "nvidia", "market", "yield", "treasury")):
            queries.append("analyst note 2025 forecast filetype:pdf")
        return queries

    def _proposed_trade(self, probability: float, liquidity: float, volume: float) -> tuple[str, float, float]:
        stake_fraction = 0.02
        confidence = 0.3
        direction = "Hold decision until research"
        if probability < 0.4:
            direction = "Lean YES after research"
        if probability > 0.6:
            direction = "Lean NO after research"
        if liquidity >= 5000 and volume >= 1000:
            stake_fraction = 0.08
            confidence += 0.2
        elif liquidity >= 2000:
            stake_fraction = 0.05
            confidence += 0.1
        if 0.45 <= probability <= 0.55:
            confidence += 0.05
        confidence = min(confidence, 0.8)
        return direction, confidence, stake_fraction

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

    def _apply_stake_limits(
        self, briefs: Sequence[ResearchBrief], portfolio: PortfolioSnapshot | None
    ) -> Sequence[ResearchBrief]:
        if portfolio is None:
            return briefs
        cash = max(0.0, portfolio.cash_balance)
        for brief in briefs:
            if cash <= 0:
                brief.max_stake = 0.0
                continue
            stake = cash * brief.stake_fraction
            stake = min(stake, brief.liquidity * 0.10)
            stake = max(stake, 0.0)
            brief.max_stake = stake if stake > 0 else 0.0
        return briefs

    def _attach_market_history(self, briefs: Sequence[ResearchBrief]) -> None:
        if not briefs:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        for brief in briefs:
            try:
                bets = self._client.get_market_bets(brief.market_id, limit=200)
            except PolymarketAPIError:
                brief.crowd_trend = "Trade history unavailable (API error)."
                continue
            points: List[tuple[datetime, float]] = []
            for bet in bets:
                created_ms = bet.get("createdTime")
                if created_ms is None:
                    continue
                created = datetime.fromtimestamp(float(created_ms) / 1000.0, tz=timezone.utc)
                if created < cutoff:
                    continue
                prob = bet.get("probAfter")
                if prob is None:
                    prob = bet.get("probBefore")
                if prob is None:
                    continue
                try:
                    prob_f = float(prob)
                except Exception:
                    continue
                points.append((created, prob_f))
            if not points:
                brief.crowd_trend = "No trades in the past 7 days."
                continue
            points.sort(key=lambda item: item[0])
            probs = [p for _, p in points]
            change = probs[-1] - probs[0]
            volatility = max(probs) - min(probs)
            direction = "flat"
            if change > 0.05:
                direction = f"rising +{change * 100:.1f}pp"
            elif change < -0.05:
                direction = f"falling {change * 100:.1f}pp"
            brief.crowd_change = change
            brief.crowd_volatility = volatility
            brief.crowd_samples = len(points)
            brief.crowd_trend = (
                f"{direction} over {len(points)} trades this week (vol {volatility * 100:.1f}pp)"
            )

    def _run_research(self, briefs: Sequence[ResearchBrief]) -> Sequence[BetRecommendation]:
        recommendations: List[BetRecommendation] = []
        for brief in briefs:
            summary, suggested_outcome, confidence = self._qualitative_research(brief)
            brief.research_summary = summary
            if suggested_outcome is None or confidence < 0.4:
                brief.recommended_direction = "Defer bet pending deeper research"
                brief.recommended_confidence = max(confidence, 0.1)
                brief.recommended_amount = 0.0
                continue
            allowable = max(0.0, brief.max_stake or 0.0)
            amount = allowable * min(1.0, max(confidence, 0.1))
            amount = round(amount, 2)
            brief.recommended_direction = f"Bet {suggested_outcome}"
            brief.recommended_confidence = confidence
            brief.recommended_amount = amount
            if amount <= 0:
                continue
            recommendations.append(
                BetRecommendation(
                    market_id=brief.market_id,
                    market_title=brief.market_title,
                    action=f"Buy {suggested_outcome}",
                    amount=amount,
                    rationale=summary,
                    confidence=confidence,
                )
            )
        return recommendations

    def _qualitative_research(self, brief: ResearchBrief) -> tuple[str, str | None, float]:
        title_lower = brief.market_title.lower()
        outcome: str | None = None
        confidence = 0.3
        summary_parts: List[str] = []
        if "texas bowl" in title_lower:
            summary_parts.append(
                "Sports matchup requires roster/injury intel and live odds data not accessible to this bot."
            )
            confidence = 0.2
        elif "trump" in title_lower and "elon" in title_lower:
            summary_parts.append(
                "Cutting 250k federal employees demands congressional approval and lengthy rulemaking; Elon Musk lacks federal authority."
            )
            outcome = "NO"
            confidence = 0.65
        elif "nvidia" in title_lower:
            summary_parts.append(
                "Nvidia's 2023-24 outperformance sets a high base; sustaining it in 2025 requires flawless execution despite valuation risk and new competition."
            )
            outcome = "NO"
            confidence = 0.55
        elif "war" in title_lower:
            summary_parts.append(
                "Geopolitical risk remains elevated but expert outlooks still rate a NATO/US-scale war as low probability within a year."
            )
            outcome = "NO"
            confidence = 0.45
        else:
            summary_parts.append("No internal heuristics matched; defer to fresh research.")
            confidence = 0.25
        crowd_note = self._crowd_note_text(brief)
        if crowd_note:
            summary_parts.append(crowd_note)
        summary = " ".join(summary_parts).strip()
        if outcome is not None:
            confidence = self._adjust_confidence_with_market(outcome, confidence, brief)
        return summary, outcome, confidence

    def _crowd_note_text(self, brief: ResearchBrief) -> str:
        if not brief.crowd_trend:
            return ""
        leaning = "YES" if brief.market_probability >= 0.5 else "NO"
        return (
            f"Crowd leaning {leaning} at {brief.market_probability * 100:.0f}% and {brief.crowd_trend}."
        )

    def _adjust_confidence_with_market(self, outcome: str, confidence: float, brief: ResearchBrief) -> float:
        prob = brief.market_probability
        crowd_support = prob if outcome == "YES" else (1 - prob)
        if crowd_support < 0.2:
            confidence -= 0.25
        elif crowd_support > 0.7:
            confidence += 0.15
        change = brief.crowd_change
        if outcome == "YES":
            if change < -0.05:
                confidence -= 0.1
            elif change > 0.05:
                confidence += 0.05
        else:
            if change > 0.05:
                confidence -= 0.1
            elif change < -0.05:
                confidence += 0.05
        return min(max(confidence, 0.05), 0.9)

    def _next_action_list(
        self, steps: Sequence[str], briefs: Sequence[ResearchBrief], bets: Sequence[BetRecommendation]
    ) -> List[str]:
        updated = [step for step in steps if "Compare current portfolio" not in step]
        if bets:
            updated.insert(0, "Review and execute approved bet tickets.")
        elif briefs:
            updated.insert(0, "Prioritize research tasks for highlighted markets")
        if not updated:
            updated.append("Expand market search criteria to refresh opportunity set")
        return updated

    @staticmethod
    def _fallback_url(payload: Dict[str, Any]) -> str:
        slug = payload.get("slug") or payload.get("urlSlug") or ""
        if slug:
            return f"https://polymarket.com/market/{slug}"
        return "https://polymarket.com/"

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 1e12:
                    timestamp /= 1000.0
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text)
        except Exception:
            return None
        return None
