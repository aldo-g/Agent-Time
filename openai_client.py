"""Thin wrapper around the OpenAI Responses API for agent reasoning."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, TYPE_CHECKING

import env_loader  # noqa: F401
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - library may not be available during linting
    OpenAI = None  # type: ignore

if TYPE_CHECKING:
    from polymarket_portfolio import PortfolioSnapshot

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class OpenAIClient:
    """Convenience wrapper that reads the API key from the environment."""

    def __init__(self, *, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. `pip install openai` to continue.")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")
        self._model = model or DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key)

    def respond(self, messages: List[dict], *, model: Optional[str] = None) -> str:
        """Send a chat-style prompt and return the first text response."""
        chosen_model = model or self._model
        response = self._client.responses.create(model=chosen_model, input=messages)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        for item in response.output or []:
            if item.type != "message":
                continue
            for content in item.content:
                if getattr(content, "type", None) == "text":
                    return content.text  # type: ignore[return-value]
        raise RuntimeError("OpenAI response did not contain text output.")


def _format_position_line(position: object) -> str:
    question = getattr(position, "question", "Unknown market")
    outcome = getattr(position, "outcome", "Unknown outcome")
    shares = getattr(position, "shares", None)
    mark_price = getattr(position, "mark_price", None)
    avg_price = getattr(position, "avg_price", None)
    price = mark_price if isinstance(mark_price, (int, float)) else None
    if price is None and isinstance(avg_price, (int, float)):
        price = avg_price
    shares_text = f"{shares:.2f}" if isinstance(shares, (int, float)) else "?"
    line = f"- {question} [{outcome}] {shares_text} shares"
    if isinstance(price, (int, float)):
        line += f" @ {price * 100:.1f}%"
        if isinstance(shares, (int, float)):
            value = price * shares
            line += f" (~${value:,.2f})"
    return line


def summarize_markets(markets: Iterable[object], portfolio: "PortfolioSnapshot | None" = None) -> str:
    """Use OpenAI to summarize interesting markets for the agent."""
    client = OpenAIClient()
    markets = list(markets)
    description_lines = []
    sports_or_crypto = False
    for idx, market in enumerate(markets):
        if idx >= 10:
            break
        title = getattr(market, "event_title", "Untitled event")
        question = getattr(market, "question", title)
        outcomes = getattr(market, "outcomes", []) or []
        tags = [str(tag).lower() for tag in getattr(market, "tags", [])]
        if any("sport" in tag or "crypto" in tag for tag in tags):
            sports_or_crypto = True
        probs = ", ".join(
            f"{getattr(outcome, 'name', 'Outcome')} {getattr(outcome, 'price', 0.0):.2f}" for outcome in outcomes
        )
        description_lines.append(f"- {title}: {question} [{probs}]")
    prompt = "\n".join(description_lines) or "No markets available."
    portfolio_lines: List[str] = []
    if portfolio and getattr(portfolio, "positions", None):
        for position in getattr(portfolio, "positions")[:5]:
            portfolio_lines.append(_format_position_line(position))
        summary_bits = []
        cash_balance = getattr(portfolio, "cash_balance", None)
        if isinstance(cash_balance, (int, float)):
            summary_bits.append(f"cash ~${cash_balance:,.2f}")
        realized = getattr(portfolio, "realized_pnl", None)
        if isinstance(realized, (int, float)):
            summary_bits.append(f"realized PnL ~${realized:,.2f}")
        unrealized = getattr(portfolio, "unrealized_pnl", None)
        if isinstance(unrealized, (int, float)):
            summary_bits.append(f"unrealized PnL ~${unrealized:,.2f}")
        if summary_bits:
            portfolio_lines.append("Ledger: " + ", ".join(summary_bits))
    user_content = (
        "Objective: maximize profit on Polymarket while respecting risk constraints. "
        "You only run once every 24 hours, so decisions must rely solely on the information you see now—"
        "plan trades as if no intraday monitoring or adjustments are possible. "
        "Choose the most promising markets to act on, referencing edge, liquidity, "
        "and how they complement (or hedge) current positions.\n\n"
        f"Markets with current probabilities:\n{prompt}"
    )
    if portfolio_lines:
        user_content += "\n\nCurrent portfolio snapshot:\n" + "\n".join(portfolio_lines)
    user_content += (
        "\n\nFor each recommended market, explain the thesis, expected value, "
        "and how you'd size or stage into it. If you need fresh news or data, add lines like "
        "'SEARCH: key terms' at the end of your reply; the runner will execute those DuckDuckGo searches "
        "after receiving your plan."
    )
    if sports_or_crypto:
        user_content = (
            "Sports or crypto markets detected. Reply only with 'Sports/crypto recommendations disabled.'"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are Agent-Time, an autonomous prediction-market operator. "
                "Your sole job is to make money on Polymarket and beat competing LLM trading agents. "
                "Study market structure, spot mispricings, cross-check existing exposure, and request "
                "further evidence (e.g., news or research queries) when needed."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return client.respond(messages)
