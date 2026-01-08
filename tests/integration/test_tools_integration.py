import os
import unittest
import urllib.error

from agent.manifold.data import load_open_markets
from agent.tools import manifold as manifold_tools
from agent.tools import web as web_tools


RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"
HAS_MANIFOLD_KEY = bool(os.environ.get("MANIFOLD_API_KEY"))
MANIFOLD_TEST_MARKET_ID = os.environ.get("MANIFOLD_TEST_MARKET_ID")


class TestToolsIntegration(unittest.TestCase):
    @unittest.skipUnless(RUN_INTEGRATION, "Set RUN_INTEGRATION=1 to enable integration tests.")
    def test_manifold_readonly_flow(self) -> None:
        if MANIFOLD_TEST_MARKET_ID:
            market_id = MANIFOLD_TEST_MARKET_ID
        else:
            events = load_open_markets(limit=1, offset=0)
            if not events:
                raise unittest.SkipTest("No open markets returned from Manifold.")
            market_id = events[0].markets[0].market_id
            if not market_id:
                raise unittest.SkipTest("Market response missing market_id.")

        output = manifold_tools._run_fetch_markets(limit=1, offset=0)
        print("\n[manifold_markets]\n", output)

        if not HAS_MANIFOLD_KEY:
            print("\n[manifold_market_details]\n SKIPPED: MANIFOLD_API_KEY not set.")
            print("\n[limit_order_preview]\n SKIPPED: MANIFOLD_API_KEY not set.")
            print("\n[event_timer]\n SKIPPED: MANIFOLD_API_KEY not set.")
            print("\n[manifold_market_history]\n SKIPPED: MANIFOLD_API_KEY not set.")
            print("\n[manifold_portfolio]\n SKIPPED: MANIFOLD_API_KEY not set.")
            return

        output = manifold_tools._run_market_details(market_id)
        print("\n[manifold_market_details]\n", output)

        # Default to YES for binary markets; otherwise choose the first answer label.
        if "Outcome type: BINARY" in output or "Outcome type: PSEUDO_NUMERIC" in output:
            outcome = "YES"
        else:
            outcome = "Top outcome"

        output = manifold_tools._run_limit_order_preview(
            market_id=market_id,
            outcome=outcome,
            amount=10.0,
            limit_prob=0.5,
        )
        print("\n[limit_order_preview]\n", output)

        output = manifold_tools._run_event_timer(market_id)
        print("\n[event_timer]\n", output)

        output = manifold_tools._run_market_history(market_id, limit=5)
        print("\n[manifold_market_history]\n", output)

        output = manifold_tools._run_portfolio()
        print("\n[manifold_portfolio]\n", output)

        output = manifold_tools._run_portfolio_analytics(max_positions=3)
        print("\n[portfolio_analytics]\n", output)

        output = manifold_tools._run_risk_gate(
            market_id=market_id,
            outcome=outcome,
            amount=10.0,
            belief_prob=0.55,
        )
        print("\n[risk_gate]\n", output)

    @unittest.skipUnless(RUN_INTEGRATION, "Set RUN_INTEGRATION=1 to enable integration tests.")
    def test_web_tools_flow(self) -> None:
        output = web_tools._run_rss_fetch(query="market", limit=3)
        if output.strip() == "No results.":
            output = web_tools._run_rss_fetch(limit=3)
        print("\n[rss_fetch]\n", output)

        try:
            output = web_tools._run_bluesky_search("manifold markets", limit=3)
            print("\n[bluesky_search]\n", output)
        except urllib.error.HTTPError as exc:
            print(f"\n[bluesky_search]\n SKIPPED: {exc.code} {exc.reason}.")

        try:
            output = web_tools._run_web_scrape("https://www.iana.org/domains/reserved", max_chars=200)
            print("\n[web_scrape]\n", output)
        except urllib.error.HTTPError as exc:
            print(f"\n[web_scrape]\n SKIPPED: {exc.code} {exc.reason}.")

        output = web_tools._run_notebook_eval("result = 1 + 1\nprint('ok')")
        print("\n[notebook_eval]\n", output)

        if web_tools.web_search_available():
            output = web_tools._run_search("Manifold Markets", limit=3)
            print("\n[duckduckgo_search]\n", output)
        else:
            print("\n[duckduckgo_search]\n SKIPPED: duckduckgo_search not installed.")


if __name__ == "__main__":
    unittest.main()
