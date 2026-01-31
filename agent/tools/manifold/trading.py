"""Trading actions and previews for Manifold markets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, Optional

from agent.manifold.portfolio import PortfolioPosition, PortfolioSnapshot, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details, lookup_answer_id, place_bet, sell_position

from .config import CUTOFF_ISO, DEFAULT_TRADE_LOG_PATH, RESOLUTION_CUTOFF_MS, TRADE_LOG_ENV
from .errors import _is_not_found_error
from .limits import _enforce_market_limit


def _append_trade_log(entry: Dict[str, object]) -> None:
    log_path = Path(os.environ.get(TRADE_LOG_ENV, str(DEFAULT_TRADE_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")


def _resolve_market_prob(details: MarketDetails, outcome: str, answer: Optional[str]) -> Optional[float]:
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = outcome.strip().upper()
        for option in details.answers:
            if option.label.upper() == normalized:
                return option.probability
        return None
    lookup_label = answer or outcome
    if not lookup_label:
        return None
    for option in details.answers:
        if option.label.strip().lower() == lookup_label.strip().lower():
            return option.probability
    return None


def _run_place_bet(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float] = None,
    answer: Optional[str] = None,
) -> str:
    if amount <= 0:
        raise RuntimeError("amount must be positive.")
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Bet skipped: market not found ({normalized})."
        raise
    if details.close_time is None:
        raise RuntimeError("Cannot trade markets without a close date.")
    if details.close_time > RESOLUTION_CUTOFF_MS:
        raise RuntimeError(f"This market resolves after {CUTOFF_ISO}; choose an earlier market.")
    snapshot = fetch_portfolio_snapshot(None)
    if snapshot.cash_balance is not None and amount > snapshot.cash_balance + 1e-6:
        raise RuntimeError(
            f"Bet amount {amount:.2f} exceeds available balance {snapshot.cash_balance:.2f}."
        )
    target_label = outcome.strip()
    answer_id = None
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = target_label.upper()
        if normalized not in {"YES", "NO"}:
            raise RuntimeError("Binary markets only accept YES or NO outcomes.")
        target_label = normalized
    else:
        lookup_label = answer or target_label
        if not lookup_label:
            raise RuntimeError("Provide answer=<label> when betting on multi-answer markets.")
        answer_id = lookup_answer_id(details, lookup_label)
        if not answer_id and lookup_label.strip().lower().startswith("top outcome"):
            best_option = None
            best_prob = -1.0
            for option in details.answers:
                if option.answer_id is None:
                    continue
                probability = option.probability if option.probability is not None else 0.0
                if probability > best_prob:
                    best_prob = probability
                    best_option = option
            if best_option:
                answer_id = best_option.answer_id
                target_label = best_option.label
        if not answer_id:
            raise RuntimeError(f"Unable to resolve answer '{lookup_label}'. Call manifold_market_details first.")
        target_label = lookup_label
    prob_before = _resolve_market_prob(details, target_label, answer)
    try:
        receipt = place_bet(
            market_id=details.market_id,
            outcome=target_label,
            amount=amount,
            limit_prob=limit_prob,
            answer_id=answer_id,
        )
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Bet skipped: market not found ({details.market_id})."
        raise
    try:
        _append_trade_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": os.environ.get("AGENT_NAME"),
                "provider": os.environ.get("AGENT_PROVIDER"),
                "model": os.environ.get("AGENT_MODEL"),
                "wallet": snapshot.wallet,
                "action": "BUY",
                "market_id": details.market_id,
                "market_slug": details.slug,
                "market": details.question,
                "market_url": details.url,
                "outcome": target_label,
                "amount": receipt.amount,
                "shares": receipt.shares,
                "prob_before": prob_before,
                "prob_after": receipt.probability,
                "limit_prob": limit_prob,
                "bet_id": receipt.bet_id,
                "status": "OPEN",
            }
        )
    except Exception:
        pass
    limit_note = f" with limit {limit_prob * 100:.2f}%" if limit_prob is not None else ""
    summary = (
        f"Wagered {amount:.2f} MANA on '{target_label}' in market {details.market_id}{limit_note}. "
        f"Bet ID: {receipt.bet_id or 'unknown'}."
    )
    return summary


def _run_sell_position(
    *,
    market_id: str,
    outcome: str,
    shares: float,
    answer: Optional[str] = None,
) -> str:
    if shares <= 0:
        raise RuntimeError("shares must be positive.")
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
    snapshot = fetch_portfolio_snapshot(None)

    holding = None
    holding_slug: Optional[str] = None
    for position in snapshot.positions:
        if position.market_id == normalized or position.slug == normalized:
            holding = position
            holding_slug = position.slug
            break

    details = None
    details_error: RuntimeError | None = None
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        details_error = exc
        if not _is_not_found_error(exc):
            raise
    if details is None and holding_slug:
        try:
            details = fetch_market_details(holding_slug)
        except RuntimeError as exc:
            if not _is_not_found_error(exc):
                raise
            details_error = details_error or exc
    if details is None:
        return f"Sell skipped: market not found ({normalized}). Details error: {details_error}"

    target_label = outcome.strip()
    answer_id = None
    outcome_type = details.outcome_type.upper()
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        normalized = target_label.upper()
        if normalized not in {"YES", "NO"}:
            raise RuntimeError("Binary markets only accept YES or NO outcomes.")
        target_label = normalized
    else:
        lookup_label = answer or target_label
        if not lookup_label:
            raise RuntimeError("Provide answer=<label> when selling multi-answer markets.")
        answer_id = lookup_answer_id(details, lookup_label)
        if not answer_id:
            raise RuntimeError(f"Unable to resolve answer '{lookup_label}'. Call manifold_market_details first.")
        target_label = lookup_label

    if holding is None:
        for position in snapshot.positions:
            if position.market_id != details.market_id and position.slug != details.slug:
                continue
            if position.outcome.strip().lower() == target_label.strip().lower():
                holding = position
                break

    available_shares = 0.0
    for position in snapshot.positions:
        if position.market_id != details.market_id and position.slug != details.slug:
            continue
        if position.outcome.strip().lower() != target_label.strip().lower():
            continue
        available_shares = max(available_shares, abs(position.shares))
    if available_shares <= 0:
        return f"Sell skipped: no holding for outcome '{target_label}' in market {details.market_id}."
    requested_shares = shares
    shares = min(shares, available_shares)
    adjusted_note = ""
    if shares < requested_shares:
        adjusted_note = f" Requested {requested_shares:.2f}, selling available {shares:.2f}."
    prob_before = _resolve_market_prob(details, target_label, answer)
    try:
        receipt = sell_position(
            market_id=details.market_id,
            outcome=target_label,
            shares=shares,
            answer_id=answer_id,
        )
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Sell skipped: market not found ({details.market_id})."
        raise
    try:
        _append_trade_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": os.environ.get("AGENT_NAME"),
                "provider": os.environ.get("AGENT_PROVIDER"),
                "model": os.environ.get("AGENT_MODEL"),
                "wallet": snapshot.wallet,
                "action": "SELL",
                "market_id": details.market_id,
                "market_slug": details.slug,
                "market": details.question,
                "market_url": details.url,
                "outcome": target_label,
                "amount": receipt.amount,
                "shares": receipt.shares if receipt.shares is not None else shares,
                "prob_before": prob_before,
                "prob_after": receipt.probability,
                "limit_prob": None,
                "bet_id": receipt.bet_id,
                "status": "OPEN",
            }
        )
    except Exception:
        pass
    executed_shares = receipt.shares if receipt.shares is not None else shares
    summary = (
        f"Sold {executed_shares:.2f} shares of '{target_label}' in market {details.market_id}. "
        f"Bet ID: {receipt.bet_id or 'unknown'}.{adjusted_note}"
    )
    return summary


def _run_limit_order_preview(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float] = None,
    answer: Optional[str] = None,
) -> str:
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
    try:
        details = fetch_market_details(normalized)
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            return f"Preview unavailable: market not found ({normalized})."
        raise
    market_prob = _resolve_market_prob(details, outcome, answer)
    lines = [
        f"Market: {details.question}",
        f"Order: {amount:.2f} on {outcome}",
    ]
    if market_prob is None:
        lines.append("Market probability unavailable; preview is approximate.")
        return "\n".join(lines)
    lines.append(f"Current implied probability: {market_prob:.2%}")
    if limit_prob is None:
        lines.append("Limit: none (market order); expected fill near current probability.")
        return "\n".join(lines)
    slippage = abs(limit_prob - market_prob)
    lines.append(f"Limit probability: {limit_prob:.2%}")
    lines.append(f"Estimated slippage vs market: {slippage:.2%}")
    if limit_prob > market_prob:
        lines.append("Limit above market: likely immediate fill.")
    elif limit_prob < market_prob:
        lines.append("Limit below market: may rest on the book until price moves.")
    else:
        lines.append("Limit at market: likely immediate fill.")
    return "\n".join(lines)
