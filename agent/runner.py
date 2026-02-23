#!/usr/bin/env python3
"""LangChain-powered daily Agent-Time trading session."""

from __future__ import annotations

import logging
import os
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict

import utils.env_loader as env_loader  # noqa: F401
from agent.callbacks import ConsoleLogger, StepDelay, ToolCallTracker
from agent.tools import build_agent_tools
from agent.tools.manifold import reset_inspected_markets, reset_portfolio_tool_state
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
DEFAULT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))
DEFAULT_STEP_DELAY_SEC = os.environ.get("AGENT_STEP_DELAY_SEC", "0")
DEFAULT_INSTRUCTION = os.environ.get(
    "AGENT_INSTRUCTION",
    (
        "Trading session: start by checking the portfolio snapshot to learn available cash and risk metrics, inspect "
        "Decide whether any currently held markets should be sold or hedged."
        " Next, scan the latest markets, run research for any non-obvious catalysts, then produce a plan to make money. Size "
        "trades prudently so the bankroll isn't overexposed in a single run. Highlight concrete trades, per-trade "
        "sizing in dollars or % bankroll, catalysts, hedge opportunities, and specific follow-up research."
    ),
)


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_step_delay() -> float:
    return max(0.0, _coerce_float(DEFAULT_STEP_DELAY_SEC, 0.0))


def _build_llm(model: str, temperature: float):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "langchain-openai is not installed. Install it with `pip install langchain-openai` "
            "and ensure you are using an OpenAI Python package version supported by LangChain."
        ) from exc
    return ChatOpenAI(model=model, temperature=temperature)


def _build_plan_prompt() -> ChatPromptTemplate:
    system_message = textwrap.dedent(
        """
        You are Agent-Time in PLAN PHASE.
        Trading tools are not available in this phase. Your job is to research and produce an execution-ready plan.
        Always start by calling `manifold_portfolio`, then `portfolio_analytics`. Use market/news tools to gather evidence.
        Call `risk_gate` for each proposed BUY action and include the same belief_prob and market_prob you intend to execute.

        Output MUST be valid JSON (no markdown fences) with this schema:
        {{
          "summary": "<short planning summary>",
          "actions": [
            {{
              "action": "place_bet" | "sell_position",
              "market_id": "<id-or-slug>",
              "outcome": "<YES/NO or answer label>",
              "amount": <number, required for place_bet>,
              "shares": <number|null, optional for sell_position>,
              "belief_prob": <number|null, required for place_bet>,
              "market_prob": <number|null>,
              "limit_prob": <number|null>,
              "answer": "<string|null>",
              "requires_news_catalyst": <true|false>,
              "catalyst_urls": ["<url>", "..."],
              "reason": "<concise rationale>"
            }}
          ],
          "no_trade_reason": "<string, required when actions is empty>"
        }}

        Rules:
        - Include only actionable trades; no placeholders.
        - Include catalyst_urls when relevant to the thesis.
        - If no trade qualifies, return actions=[] with no_trade_reason.
        - Make exactly one tool call at a time.
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


def _build_execution_prompt() -> ChatPromptTemplate:
    system_message = textwrap.dedent(
        """
        You are Agent-Time in EXECUTION PHASE.
        Use only available trade tools to execute the approved plan. Do not invent new trades and do not perform research.
        For each planned action, attempt exactly one tool call (or skip with reason if the action is invalid).
        If a tool returns a skip/error message, report it and continue to the next planned action.

        Final output format:
        1) "Summary - <what was executed or skipped>"
        2) For each executed trade:
           "Trade - <executed action>"
           "Reason - <why executed>"
        3) For each skipped planned action:
           "Skipped - <action>"
           "Reason - <tool/message>"
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
    model: str,
    temperature: float,
    max_steps: int,
    *,
    tools,
    prompt: ChatPromptTemplate,
    react_phase_rules: str,
) -> AgentExecutor:
    llm = _build_llm(model, temperature)
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
        react_prompt = (
            PromptTemplate.from_template(
                """
                You are Agent-Time using ReAct fallback.
                {phase_rules}

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
            ).partial(phase_rules=react_phase_rules)
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


def _phase_callbacks(*, verbose: bool) -> tuple[ToolCallTracker, list]:
    tracker = ToolCallTracker()
    callbacks = [tracker]
    step_delay = _resolve_step_delay()
    if step_delay > 0:
        callbacks.append(StepDelay(step_delay))
    if verbose:
        callbacks.append(ConsoleLogger())
    return tracker, callbacks


def run_daily_session(
    instruction: str,
    *,
    model: str,
    temperature: float,
    max_steps: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute an autonomous two-phase session and return the final output."""
    reset_inspected_markets()
    reset_portfolio_tool_state()
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    plan_tools = build_agent_tools(mode="plan")
    execute_tools = build_agent_tools(mode="execute")
    plan_executor = _build_agent_executor(
        model,
        temperature,
        max_steps,
        tools=plan_tools,
        prompt=_build_plan_prompt(),
        react_phase_rules=(
            "This is PLAN PHASE. Trading tools are unavailable. Research with tools, then produce JSON plan only."
        ),
    )
    execute_executor = _build_agent_executor(
        model,
        temperature,
        max_steps,
        tools=execute_tools,
        prompt=_build_execution_prompt(),
        react_phase_rules=(
            "This is EXECUTION PHASE. Use only trade tools and execute the provided plan. "
            "Do not add new trades."
        ),
    )
    plan_tracker, plan_callbacks = _phase_callbacks(verbose=verbose)
    execute_tracker, execute_callbacks = _phase_callbacks(verbose=verbose)
    plan_inputs = {
        "input": instruction,
        "chat_history": [],
    }
    try:
        plan_result = plan_executor.invoke(plan_inputs, config={"callbacks": plan_callbacks})
    except Exception:  # noqa: BLE001
        logging.exception("Plan phase failed.")
        raise
    plan_output = plan_result.get("output") if isinstance(plan_result, dict) else plan_result
    execute_input = textwrap.dedent(
        f"""
        Original objective:
        {instruction}

        Approved plan JSON from planning phase:
        {plan_output}
        """
    ).strip()
    try:
        execute_result = execute_executor.invoke(
            {"input": execute_input, "chat_history": []},
            config={"callbacks": execute_callbacks},
        )
    except Exception:  # noqa: BLE001
        logging.exception("Execution phase failed.")
        raise
    finished_at = datetime.now(timezone.utc)
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    result = execute_result if isinstance(execute_result, dict) else {"output": execute_result}
    if isinstance(result, dict):
        successful_tools = plan_tracker.successful_tools + execute_tracker.successful_tools
        failed_tools = plan_tracker.failed_tools + execute_tracker.failed_tools
        trade_outputs = plan_tracker.trade_outputs + execute_tracker.trade_outputs
        failed_tool_errors = plan_tracker.failed_tool_errors + execute_tracker.failed_tool_errors
        result["plan_output"] = plan_output
        result["tool_calls"] = successful_tools
        result["tool_calls_unique"] = sorted(set(successful_tools))
        result["tool_call_failures"] = failed_tools
        result["captured_trades"] = trade_outputs
        result["tool_call_errors"] = failed_tool_errors
        result["run_started_at"] = started_at.isoformat()
        result["run_finished_at"] = finished_at.isoformat()
        result["run_duration_ms"] = duration_ms
    return result


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
