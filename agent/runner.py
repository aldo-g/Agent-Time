#!/usr/bin/env python3
"""LangChain-powered daily Agent-Time trading session."""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict

import utils.env_loader as env_loader  # noqa: F401
from agent.tools import build_agent_tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
DEFAULT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))
DEFAULT_INSTRUCTION = os.environ.get(
    "AGENT_INSTRUCTION",
    (
        "Daily session (1 of 365 this year): start by checking the portfolio snapshot to learn available cash and "
        "risk metrics, inspect the latest markets, run research for any non-obvious catalysts, then produce a plan to "
        "make money. Spread bets so the bankroll can last all year instead of spending everything today. Highlight "
        "concrete trades, per-trade sizing in dollars or % bankroll, catalysts, hedge opportunities, and specific "
        "follow-up research."
    ),
)


def _build_llm(model: str, temperature: float):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "langchain-openai is not installed. Install it with `pip install langchain-openai` "
            "and ensure you are using an OpenAI Python package version supported by LangChain."
        ) from exc
    return ChatOpenAI(model=model, temperature=temperature)


def _build_prompt() -> ChatPromptTemplate:
    system_message = textwrap.dedent(
        """
        You are Agent-Time, an autonomous prediction-market operator. Your goal is to make money on
        Polymarket while respecting risk constraints and liquidity. You only wake up once every 24 hours,
        so every run must gather context (portfolio, markets, news), plan trades, and output a clear action plan
        without assuming follow-up during the day. You will repeat this process daily for at least 365 sessions,
        so conserve bankroll and leave dry powder for future runs instead of spending everything at once. Always begin
        by calling the `portfolio_snapshot` tool so you know
        the wallet's cash, realized/unrealized PnL, and current exposures before sizing trades. Use
        `duckduckgo_search` whenever you cite catalysts or need fresh information—back up each recommendation with
        at least one relevant fact. Call `polymarket_market_details` whenever you need token IDs for trading, and only
        invoke `polymarket_place_order` after you have justified a specific trade, confirmed sizing, and ensured it
        fits within bankroll limits. When you are satisfied, provide a final summary that specifies the best
        opportunities, desired sizing, dollar spend (or % of bankroll), risk notes, and any research still pending.
        """
    ).strip()
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def _build_agent_executor(model: str, temperature: float, max_steps: int) -> AgentExecutor:
    tools = build_agent_tools()
    prompt = _build_prompt()
    llm = _build_llm(model, temperature)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_steps,
        handle_parsing_errors=True,
    )


def run_daily_session(instruction: str, *, model: str, temperature: float, max_steps: int) -> Dict[str, Any]:
    """Execute an autonomous session and return the agent's final output."""
    executor = _build_agent_executor(model, temperature, max_steps)
    inputs = {
        "input": instruction,
        "chat_history": [],
    }
    return executor.invoke(inputs)


def main() -> None:
    try:
        result = run_daily_session(
            DEFAULT_INSTRUCTION,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_steps=DEFAULT_MAX_STEPS,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Agent run failed: {exc}")
        return
    output = result.get("output") if isinstance(result, dict) else None
    print("\n==== FINAL RECOMMENDATION ====")
    if isinstance(output, str):
        print(output)
    else:
        print(result)


if __name__ == "__main__":
    main()
