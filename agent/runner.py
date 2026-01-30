#!/usr/bin/env python3
"""LangChain-powered daily Agent-Time trading session."""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict

import utils.env_loader as env_loader  # noqa: F401
from agent.callbacks import ConsoleLogger, ToolCallTracker
from agent.tools import build_agent_tools
from agent.tools.manifold import reset_inspected_markets
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "openai").lower()
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
DEFAULT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))
DEFAULT_INSTRUCTION = os.environ.get(
    "AGENT_INSTRUCTION",
    (
        "Trading session: start by checking the portfolio snapshot to learn available cash and risk metrics, inspect "
        "the latest markets, run research for any non-obvious catalysts, then produce a plan to make money. Size "
        "trades prudently so the bankroll isn't overexposed in a single run. Highlight concrete trades, per-trade "
        "sizing in dollars or % bankroll, catalysts, hedge opportunities, and specific follow-up research."
    ),
)


def _ensure_provider_env(provider: str) -> None:
    """Backfill provider-specific API key variables if aliases were supplied."""
    aliases = {
        "anthropic": [("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")],
        "claude": [("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")],
        "google": [("GOOGLE_API_KEY", "GEMINI_API_KEY")],
        "gemini": [("GOOGLE_API_KEY", "GEMINI_API_KEY")],
    }
    target_aliases = aliases.get(provider.lower())
    if not target_aliases:
        return
    for target, alias in target_aliases:
        if os.environ.get(target):
            continue
        alias_value = os.environ.get(alias)
        if alias_value:
            os.environ[target] = alias_value


def _build_llm(model: str, temperature: float, provider: str):
    normalized = provider.lower()
    _ensure_provider_env(normalized)
    if normalized in {"openai", "gpt", "chatgpt"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-openai is not installed. Install it with `pip install langchain-openai` "
                "and ensure you are using an OpenAI Python package version supported by LangChain."
            ) from exc
        return ChatOpenAI(model=model, temperature=temperature)
    if normalized in {"anthropic", "claude"}:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-anthropic is not installed. Install it with `pip install langchain-anthropic`."
            ) from exc
        return ChatAnthropic(model=model, temperature=temperature)
    if normalized in {"google", "gemini"}:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-google-genai is not installed. Install it with `pip install langchain-google-genai`."
            ) from exc
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"Unsupported LLM provider '{provider}'.")


def _build_prompt() -> ChatPromptTemplate:
    system_message = textwrap.dedent(
        """
        You are Agent-Time, an autonomous prediction-market operator. Your goal is to make money on
        Manifold with play-money Mana while respecting risk constraints and liquidity. Each run must gather context
        (portfolio, markets, news), plan trades, and output a clear action plan without assuming near-term follow-up.
        Conserve bankroll and avoid overbetting in any single run so you can keep trading over time. Always begin
        by calling the `manifold_portfolio` tool so you know the account's cash, realized/unrealized PnL, and current
        exposures before sizing trades. Check market close times and resolution criteria before trading, but you may trade
        any market that fits your thesis. Use `duckduckgo_search` whenever you cite catalysts or need fresh information—back up
        each recommendation with at least one relevant fact. Call `manifold_market_details` whenever you need the full set of
        answers or odds for a market, and once you have justified a trade (including bankroll checks and catalysts) immediately call
        `manifold_place_bet` to execute it before moving on. Make exactly one tool call at a time, then wait for its result before
        issuing another call—never batch or request multiple tools simultaneously. Do not leave actionable trades as suggestions—either submit
        them or explain why they were rejected. If you make no trades, state a clear reason using a line that begins with "No-Trade Reason -".
        When you are satisfied, provide a final summary with the following format:
        1) A short paragraph beginning with "Summary -" describing what was accomplished.
        2) For each executed trade, include lines formatted exactly as "Trade - <market and action>" and on the next line "Reason - <concise justification>".
        Mention remaining cash or pending research after the trade list if relevant.
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


def _build_agent_executor(
    model: str, temperature: float, provider: str, max_steps: int, verbose: bool
) -> AgentExecutor:
    tools = build_agent_tools()
    prompt = _build_prompt()
    llm = _build_llm(model, temperature, provider)
    agent = None
    tool_agent_error: Exception | None = None
    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
    except NotImplementedError:
        tool_agent_error = None
        agent = None
    except Exception as exc:  # pragma: no cover - defensive fallback for providers without tool support
        tool_agent_error = exc
        agent = None
    if agent is None:
        # Some provider clients may not support tool binding; fall back to a ReAct-style agent
        # with an explicit ReAct-format prompt.
        react_prompt = PromptTemplate.from_template(
            """
            You are Agent-Time, an autonomous Manifold trading agent. Use the tools below to gather context and place trades.

            You have access to the following tools:
            {tools}

            Use this format:
            Question: the input you must solve
            Thought: your reasoning
            Action: the tool to use, one of [{tool_names}]
            Action Input: the input to that tool
            Observation: the tool result
            ...(repeat Thought/Action/Action Input/Observation as needed)...
            Thought: I now have the final answer
            Final Answer: your concise result

            Question: {input}
            {agent_scratchpad}
            """
        )
        agent = create_react_agent(llm, tools, react_prompt)
        if tool_agent_error:
            print(f"Falling back to ReAct agent because tool-calling agent failed: {tool_agent_error}")
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # use our own lightweight logger to avoid StdOut handler issues
        max_iterations=max_steps,
        handle_parsing_errors=True,
    )


def run_daily_session(
    instruction: str,
    *,
    model: str,
    provider: str,
    temperature: float,
    max_steps: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute an autonomous session and return the agent's final output."""
    reset_inspected_markets()
    executor = _build_agent_executor(model, temperature, provider, max_steps, verbose)
    inputs = {
        "input": instruction,
        "chat_history": [],
    }
    tracker = ToolCallTracker()
    callbacks = [tracker]
    if verbose:
        callbacks.append(ConsoleLogger())
    result = executor.invoke(inputs, config={"callbacks": callbacks})
    if isinstance(result, dict):
        result["tool_calls"] = tracker.successful_tools
        result["tool_calls_unique"] = sorted(set(tracker.successful_tools))
        result["tool_call_failures"] = tracker.failed_tools
        result["captured_trades"] = tracker.trade_outputs
        result["tool_call_errors"] = tracker.failed_tool_errors
    return result


def main() -> None:
    try:
        result = run_daily_session(
            DEFAULT_INSTRUCTION,
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
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
