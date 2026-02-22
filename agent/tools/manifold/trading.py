"""Trading actions and previews for Manifold markets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, Optional
import urllib.parse

from agent.manifold.portfolio import PortfolioSnapshot, fetch_portfolio_snapshot
from agent.manifold.trading import MarketDetails, fetch_market_details, lookup_answer_id, place_bet, sell_position

from . import config
from .config import DEFAULT_TRADE_LOG_PATH, TRADE_LOG_ENV
from .errors import _extract_sell_cap_shares, _is_no_position_error, _is_not_found_error
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
        if option.answer_id and option.answer_id == lookup_label:
            return option.probability
        if option.label.strip().lower() == lookup_label.strip().lower():
            return option.probability
    return None


def _estimate_bankroll(snapshot: PortfolioSnapshot) -> tuple[float, float]:
    cash = snapshot.cash_balance or 0.0
    positions_value = snapshot.investment_value
    if positions_value is None and snapshot.unrealized_pnl is not None:
        positions_value = float(snapshot.unrealized_pnl)
    if positions_value is not None:
        positions_value = float(positions_value)
        return cash + positions_value, abs(positions_value)
    net_value = 0.0
    gross_exposure = 0.0
    for position in snapshot.positions:
        value = position.estimated_value()
        if value is None:
            continue
        net_value += value
        gross_exposure += abs(value)
    return cash + net_value, gross_exposure


def _estimate_position_exposure(
    snapshot: PortfolioSnapshot,
    *,
    market_id: str,
    outcome_label: str,
) -> float:
    exposure = 0.0
    target = outcome_label.strip().lower()
    for position in snapshot.positions:
        if position.market_id != market_id:
            continue
        if position.outcome.strip().lower() != target:
            continue
        value = position.estimated_value()
        if value is None:
            fallback_price = position.mark_price if position.mark_price is not None else position.avg_price
            if fallback_price is None:
                continue
            value = fallback_price * position.shares
        exposure += abs(value)
    return exposure


def _normalize_domain(url: str) -> str | None:
    raw = (url or "").strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or parsed.path).strip().lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _is_trusted_source(url: str) -> bool:
    domain = _normalize_domain(url)
    if not domain:
        return False
    for trusted in config.TRUSTED_CATALYST_DOMAINS:
        if domain == trusted or domain.endswith(f".{trusted}"):
            return True
    return False


def _run_place_bet(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    belief_prob: Optional[float] = None,
    market_prob: Optional[float] = None,
    limit_prob: Optional[float] = None,
    answer: Optional[str] = None,
    requires_news_catalyst: bool = True,
    catalyst_urls: Optional[list[str]] = None,
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
    snapshot = fetch_portfolio_snapshot(None, api_key=os.environ.get("MANIFOLD_API_KEY"))
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
            labels = [option.label for option in details.answers if option.label]
            preview = ", ".join(labels[:5])
            more = f" (+{len(labels) - 5} more)" if len(labels) > 5 else ""
            return (
                f"Bet skipped: unable to resolve answer '{lookup_label}' for market {details.market_id}. "
                f"Valid answers: {preview}{more}. Call manifold_market_details and pass answer=<label>."
            )
        target_label = lookup_label
    prob_before = market_prob
    if prob_before is None:
        prob_before = _resolve_market_prob(details, target_label, answer_id or answer)

    if belief_prob is None:
        return (
            "Bet skipped: belief_prob is required for manifold_place_bet execution guard. "
            "Run risk_gate first and pass belief_prob."
        )
    try:
        belief_prob = float(belief_prob)
    except Exception:
        return "Bet skipped: belief_prob must be numeric between 0 and 1."
    if not 0.0 < belief_prob < 1.0:
        return "Bet skipped: belief_prob must be between 0 and 1."
    if prob_before is None:
        return (
            f"Bet skipped: market probability unavailable for outcome '{target_label}' in market {details.market_id}. "
            "Cannot enforce edge guard."
        )
    edge = belief_prob - prob_before
    if edge <= 0:
        return (
            f"Bet skipped: non-positive edge ({edge:.2%}). "
            "Execution guard blocks trades when belief_prob <= market_prob."
        )

    if requires_news_catalyst and config.REQUIRE_TRUSTED_CATALYST_FOR_NEWS_TRADES:
        urls = [url for url in (catalyst_urls or []) if str(url).strip()]
        if not urls:
            return (
                "Bet skipped: news-driven trade requires catalyst_urls with at least one trusted source "
                f"({', '.join(config.TRUSTED_CATALYST_DOMAINS[:3])}, ...)."
            )
        trusted = [url for url in urls if _is_trusted_source(url)]
        if not trusted:
            provided_domains = sorted(
                {
                    domain
                    for domain in (_normalize_domain(url) for url in urls)
                    if domain
                }
            )
            provided_preview = ", ".join(provided_domains) if provided_domains else "unknown"
            return (
                "Bet skipped: catalyst sources are not trusted. "
                f"Provided domains: {provided_preview}. "
                "Add at least one trusted source URL."
            )

    bankroll, gross_exposure = _estimate_bankroll(snapshot)
    if bankroll <= 0:
        return "Bet skipped: bankroll unavailable or non-positive; execution guard cannot size safely."
    max_bet = bankroll * config.RISK_MAX_BET_PCT
    if amount > max_bet + 1e-6:
        return (
            f"Bet skipped: amount {amount:.2f} exceeds risk cap {config.RISK_MAX_BET_PCT:.0%} "
            f"of bankroll ({max_bet:.2f})."
        )
    kelly_cap = bankroll * max(0.0, edge * config.KELLY_MULTIPLIER)
    if amount > kelly_cap + 1e-6:
        return (
            f"Bet skipped: amount {amount:.2f} exceeds Kelly cap {kelly_cap:.2f} "
            f"(edge {edge:.2%}, multiplier {config.KELLY_MULTIPLIER:.2f})."
        )
    max_gross = bankroll * config.RISK_MAX_GROSS_EXPOSURE_PCT
    if gross_exposure + amount > max_gross + 1e-6:
        return (
            f"Bet skipped: gross exposure cap exceeded ({gross_exposure + amount:.2f} > {max_gross:.2f})."
        )
    current_position_exposure = _estimate_position_exposure(
        snapshot,
        market_id=details.market_id,
        outcome_label=target_label,
    )
    max_single = bankroll * config.RISK_MAX_SINGLE_POSITION_PCT
    if current_position_exposure + amount > max_single + 1e-6:
        return (
            "Bet skipped: single-position cap exceeded "
            f"({current_position_exposure + amount:.2f} > {max_single:.2f})."
        )

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
                "belief_prob": belief_prob,
                "edge": edge,
                "limit_prob": limit_prob,
                "requires_news_catalyst": requires_news_catalyst,
                "trusted_catalyst_urls": [url for url in (catalyst_urls or []) if _is_trusted_source(url)],
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
    shares: float | None = None,
    answer: Optional[str] = None,
) -> str:
    allowed, normalized, msg = _enforce_market_limit(market_id)
    if not allowed:
        return msg or "Market limit reached."
    snapshot = fetch_portfolio_snapshot(None, api_key=os.environ.get("MANIFOLD_API_KEY"))

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
    epsilon = 1e-6
    if available_shares <= epsilon:
        return f"Sell skipped: no holding for outcome '{target_label}' in market {details.market_id}."
    requested_shares = shares
    shares = max(0.0, available_shares)
    adjusted_note = ""
    if requested_shares is not None and abs(shares - requested_shares) > epsilon:
        adjusted_note = f" Requested {requested_shares:.2f}, selling full position {shares:.2f}."
    prob_before = _resolve_market_prob(details, target_label, answer)
    submitted_shares = shares
    try:
        receipt = sell_position(
            market_id=details.market_id,
            outcome=target_label,
            shares=submitted_shares,
            answer_id=answer_id,
        )
    except RuntimeError as exc:
        capped_shares = _extract_sell_cap_shares(exc)
        if capped_shares is not None:
            if capped_shares <= epsilon:
                return f"Sell skipped: no sellable shares for outcome '{target_label}' in market {details.market_id}."
            submitted_shares = capped_shares
            receipt = sell_position(
                market_id=details.market_id,
                outcome=target_label,
                shares=submitted_shares,
                answer_id=answer_id,
            )
            if abs(submitted_shares - shares) > epsilon:
                adjusted_note = (
                    f"{adjusted_note} Live position capped at {submitted_shares:.8f} shares before execution."
                )
        elif _is_no_position_error(exc):
            return f"Sell skipped: no holding for outcome '{target_label}' in market {details.market_id}."
        elif _is_not_found_error(exc):
            return f"Sell skipped: market not found ({details.market_id})."
        else:
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
                "shares": receipt.shares if receipt.shares is not None else submitted_shares,
                "prob_before": prob_before,
                "prob_after": receipt.probability,
                "limit_prob": None,
                "bet_id": receipt.bet_id,
                "status": "OPEN",
            }
        )
    except Exception:
        pass
    executed_shares = receipt.shares if receipt.shares is not None else submitted_shares
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
