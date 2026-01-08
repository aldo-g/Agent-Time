import unittest
from unittest.mock import patch
from datetime import datetime, timezone

from agent.manifold.data import EventSummary, MarketSummary, OutcomeQuote
from agent.manifold.history import MarketBet
from agent.manifold.portfolio import PortfolioPosition, PortfolioSnapshot
from agent.manifold.trading import BetReceipt, MarketDetails, OutcomeOption
from agent.manifold.constants import RESOLUTION_CUTOFF_MS
from agent.tools import manifold as tools


class TestManifoldTools(unittest.TestCase):
    def test_run_fetch_markets_uses_cache(self) -> None:
        event = EventSummary(
            event_id="event-1",
            title="Election Odds",
            url="https://example.com/markets/1",
            tags=["politics"],
            markets=[
                MarketSummary(
                    event_id="event-1",
                    event_title="Election Odds",
                    market_id="m1",
                    question="Will X win?",
                    url="https://example.com/markets/1",
                    outcomes=[OutcomeQuote(name="YES", price=0.62)],
                    tags=["politics"],
                )
            ],
        )
        with patch("agent.tools.manifold._load_cached_markets", return_value=[event]):
            output = tools._run_fetch_markets(limit=1, offset=0)
        self.assertIn("Election Odds", output)
        self.assertIn("Will X win?", output)

    def test_run_portfolio_summary(self) -> None:
        snapshot = PortfolioSnapshot(
            wallet="alice",
            cash_balance=120.0,
            realized_pnl=5.0,
            unrealized_pnl=2.0,
            positions=[
                PortfolioPosition(
                    market_id="m1",
                    question="Will it rain?",
                    outcome="YES",
                    shares=10.0,
                    avg_price=0.4,
                    mark_price=0.5,
                    pnl=1.0,
                )
            ],
        )
        with patch("agent.tools.manifold.fetch_portfolio_snapshot", return_value=snapshot):
            output = tools._run_portfolio()
        self.assertIn("Wallet: alice", output)
        self.assertIn("Top positions:", output)

    def test_run_market_details(self) -> None:
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url="https://example.com/m1",
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[OutcomeOption(outcome="YES", label="YES", probability=0.55, answer_id="a1")],
            close_time=None,
            raw={},
        )
        with patch("agent.tools.manifold.fetch_market_details", return_value=details):
            output = tools._run_market_details("m1")
        self.assertIn("Market m1 details:", output)
        self.assertIn("Available outcomes:", output)
        self.assertIn("YES", output)

    def test_run_place_bet(self) -> None:
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[OutcomeOption(outcome="YES", label="YES", probability=0.55)],
            close_time=RESOLUTION_CUTOFF_MS - 1000,
            raw={},
        )
        snapshot = PortfolioSnapshot(wallet="alice", cash_balance=100.0)
        receipt = BetReceipt(bet_id="bet123", outcome="YES", amount=10.0, shares=None, probability=None, response={})
        with patch("agent.tools.manifold.fetch_market_details", return_value=details), patch(
            "agent.tools.manifold.fetch_portfolio_snapshot", return_value=snapshot
        ), patch("agent.tools.manifold.place_bet", return_value=receipt):
            output = tools._run_place_bet(market_id="m1", outcome="YES", amount=10.0)
        self.assertIn("Wagered 10.00 MANA", output)
        self.assertIn("Bet ID: bet123", output)

    def test_run_sell_position(self) -> None:
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[OutcomeOption(outcome="YES", label="YES", probability=0.55)],
            close_time=None,
            raw={},
        )
        snapshot = PortfolioSnapshot(
            wallet="alice",
            positions=[
                PortfolioPosition(
                    market_id="m1",
                    question="Will it snow?",
                    outcome="YES",
                    shares=5.0,
                )
            ],
        )
        receipt = BetReceipt(bet_id="bet456", outcome="YES", amount=0.0, shares=2.0, probability=None, response={})
        with patch("agent.tools.manifold.fetch_market_details", return_value=details), patch(
            "agent.tools.manifold.fetch_portfolio_snapshot", return_value=snapshot
        ), patch("agent.tools.manifold.sell_position", return_value=receipt):
            output = tools._run_sell_position(market_id="m1", outcome="YES", shares=2.0)
        self.assertIn("Sold 2.00 shares", output)
        self.assertIn("Bet ID: bet456", output)

    def test_run_limit_order_preview(self) -> None:
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[
                OutcomeOption(outcome="YES", label="YES", probability=0.4),
                OutcomeOption(outcome="NO", label="NO", probability=0.6),
            ],
            close_time=None,
            raw={},
        )
        with patch("agent.tools.manifold.fetch_market_details", return_value=details):
            output = tools._run_limit_order_preview(market_id="m1", outcome="YES", amount=5.0, limit_prob=0.5)
        self.assertIn("Limit probability: 50.00%", output)

    def test_run_portfolio_analytics(self) -> None:
        snapshot = PortfolioSnapshot(
            wallet="alice",
            cash_balance=100.0,
            positions=[
                PortfolioPosition(
                    market_id="m1",
                    question="Will it snow?",
                    outcome="YES",
                    shares=10.0,
                    mark_price=0.1,
                )
            ],
        )
        with patch("agent.tools.manifold.fetch_portfolio_snapshot", return_value=snapshot):
            output = tools._run_portfolio_analytics(max_positions=1)
        self.assertIn("Risk alerts: none.", output)

    def test_run_event_timer(self) -> None:
        now = datetime.now(timezone.utc)
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[OutcomeOption(outcome="YES", label="YES", probability=0.55)],
            close_time=int(now.timestamp() * 1000) + 3600 * 1000,
            raw={},
        )
        with patch("agent.tools.manifold.fetch_market_details", return_value=details):
            output = tools._run_event_timer("m1")
        self.assertIn("OPEN", output)

    def test_run_risk_gate(self) -> None:
        snapshot = PortfolioSnapshot(wallet="alice", cash_balance=1000.0, positions=[])
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[
                OutcomeOption(outcome="YES", label="YES", probability=0.5),
                OutcomeOption(outcome="NO", label="NO", probability=0.5),
            ],
            close_time=None,
            raw={},
        )
        with patch("agent.tools.manifold.fetch_portfolio_snapshot", return_value=snapshot), patch(
            "agent.tools.manifold.fetch_market_details", return_value=details
        ):
            output = tools._run_risk_gate(
                market_id="m1",
                outcome="YES",
                amount=10.0,
                belief_prob=0.6,
            )
        self.assertIn("Risk gate: PASS", output)
        self.assertIn("Kelly-style cap", output)

    def test_run_market_history(self) -> None:
        details = MarketDetails(
            market_id="m1",
            slug=None,
            url=None,
            question="Will it snow?",
            outcome_type="BINARY",
            answers=[OutcomeOption(outcome="YES", label="YES", probability=0.5)],
            close_time=None,
            raw={},
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        bets = [
            MarketBet(timestamp=now_ms, outcome="YES", amount=10.0, prob_after=0.52),
            MarketBet(timestamp=now_ms - 1000, outcome="NO", amount=5.0, prob_after=0.51),
        ]
        with patch("agent.tools.manifold.fetch_market_details", return_value=details), patch(
            "agent.tools.manifold.fetch_market_history", return_value=bets
        ):
            output = tools._run_market_history("m1", limit=2)
        self.assertIn("Recent bets analyzed: 2", output)


if __name__ == "__main__":
    unittest.main()
