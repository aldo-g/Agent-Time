#!/usr/bin/env python3
"""LangChain-powered daily Agent-Time trading session."""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Any, Dict

import utils.env_loader as env_loader  # noqa: F401
from agent.tools import build_agent_tools
from agent.callbacks import ConsoleLogger, ToolCallTracker
from agent.tools.manifold import reset_inspected_markets
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))


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
        Manifold with play-money Mana while respecting risk constraints and liquidity. Each run must gather context
        (portfolio, markets, news), plan trades, and output a clear action plan without assuming immediate follow-up. Use the
        available tools to research markets, inspect existing exposure, and request additional information when needed. Check
        market close times and resolution criteria before trading. When you commit to a trade, submit it via `manifold_place_bet`
        right after the justification so that recommendations are actually executed. When you are satisfied, provide a final summary using this format:
        1) Start with "Summary -" and briefly recap what was done.
        2) For each executed trade, list two lines: "Trade - <market/action>" followed by "Reason - <justification>".
        Mention bankroll status or follow-ups after the trade list when relevant.
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


def _build_agent_executor(model: str, temperature: float, max_steps: int, verbose: bool) -> AgentExecutor:
    tools = build_agent_tools()
    prompt = _build_prompt()
    llm = _build_llm(model, temperature)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_steps,
        handle_parsing_errors=True,
    )


def run_daily_session(
    instruction: str, *, model: str, temperature: float, max_steps: int, verbose: bool = False
) -> Dict[str, Any]:
    """Execute an autonomous session and return the agent's final output."""
    reset_inspected_markets()
    executor = _build_agent_executor(model, temperature, max_steps, verbose)
    inputs = {
        "input": instruction,
        "chat_history": [],
    }
    callbacks = [ToolCallTracker()]
    if verbose:
        callbacks.append(ConsoleLogger())
    return executor.invoke(inputs, config={"callbacks": callbacks})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction",
        default=(
            "Trading session: inspect the latest markets and the current portfolio, then produce a plan to make money. "
            "Highlight concrete trades, sizing, and catalysts. Ask for research if needed."
        ),
        help="High-level instruction passed to the agent.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: 0.2).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum tool calls/iterations (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable LangChain verbose output to show intermediate tool calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_daily_session(
            args.instruction,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            verbose=args.verbose,
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
