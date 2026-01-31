"""CLI helper to sell a Manifold position by market id or slug."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

from agent.manifold.portfolio import PortfolioPosition, fetch_portfolio_snapshot
from agent.manifold.trading import fetch_market_details, lookup_answer_id, sell_position


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sell shares on a Manifold market.")
    parser.add_argument(
        "market_id",
        help="Manifold market id or slug to sell.",
    )
    parser.add_argument(
        "--outcome",
        default=None,
        help="Outcome to sell (YES/NO for binary markets). If omitted with --all, uses the only held outcome; otherwise defaults to YES.",
    )
    parser.add_argument(
        "--shares",
        type=float,
        help="Number of shares to sell (positive).",
    )
    parser.add_argument(
        "--answer-id",
        default=None,
        help="Optional answerId for multi-answer markets.",
    )
    parser.add_argument(
        "--all",
        dest="sell_all",
        action="store_true",
        help="Sell the entire position for the chosen outcome.",
    )
    return parser.parse_args(argv)


def _resolve_shares(
    market_id: str,
    desired_outcome: str | None,
    api_key: str,
) -> Tuple[float, str]:
    """Return (shares_to_sell, resolved_outcome) for a given market."""
    snapshot = fetch_portfolio_snapshot(api_key=api_key)
    matches: list[PortfolioPosition] = []
    for position in snapshot.positions:
        if position.market_id == market_id or position.slug == market_id:
            matches.append(position)
    if not matches:
        raise RuntimeError(f"No holding found for market '{market_id}'.")
    if desired_outcome:
        for position in matches:
            if position.outcome.strip().lower() == desired_outcome.strip().lower():
                shares = abs(position.shares)
                if shares <= 0:
                    raise RuntimeError(f"Holding for {desired_outcome} is zero shares in market '{market_id}'.")
                return shares, position.outcome
        raise RuntimeError(
            f"No holding for outcome '{desired_outcome}' found in market '{market_id}'. "
            "Specify a different outcome or omit --outcome to pick the only holding."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple outcomes held in this market; specify --outcome to choose which to sell."
        )
    position = matches[0]
    shares = abs(position.shares)
    if shares <= 0:
        raise RuntimeError(f"Holding is zero shares for market '{market_id}'.")
    return shares, position.outcome


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    api_key = os.environ.get("MANIFOLD_API_KEY")
    if not api_key:
        sys.stderr.write("Error: set MANIFOLD_API_KEY in your environment before calling this CLI.\n")
        return 1
    if not args.sell_all and (args.shares is None or args.shares <= 0):
        sys.stderr.write("Error: provide --shares (>0) or --all to sell an entire holding.\n")
        return 1
    if args.sell_all and args.shares is not None:
        sys.stderr.write("Error: use either --shares or --all, not both.\n")
        return 1
    shares = args.shares
    outcome = args.outcome
    if not args.sell_all and outcome is None:
        outcome = "YES"
    if args.sell_all:
        try:
            shares, outcome = _resolve_shares(args.market_id, outcome, api_key)
        except Exception as exc:  # pragma: no cover - CLI convenience
            sys.stderr.write(f"Sell failed: {exc}\n")
            return 1
    answer_id = args.answer_id
    try:
        details = fetch_market_details(args.market_id)
        outcome_type = details.outcome_type.upper()
        if answer_id is None and outcome_type not in {"BINARY", "PSEUDO_NUMERIC"}:
            resolved = lookup_answer_id(details, outcome or "")
            if resolved is None:
                sys.stderr.write(
                    "Sell failed: outcome is not YES/NO and answerId could not be resolved automatically. "
                    "Provide --answer-id explicitly.\n"
                )
                return 1
            answer_id = resolved
    except Exception as exc:  # pragma: no cover - CLI convenience
        sys.stderr.write(f"Sell failed: unable to fetch market details ({exc}).\n")
        return 1
    try:
        receipt = sell_position(
            market_id=args.market_id,
            outcome=outcome,
            shares=shares,
            answer_id=answer_id,
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        sys.stderr.write(f"Sell failed: {exc}\n")
        return 1
    sold_shares = receipt.shares if receipt.shares is not None else shares
    sold_amount = receipt.amount if receipt.amount is not None else 0.0
    print(
        f"Sold {sold_shares:.2f} shares of '{receipt.outcome}' in {args.market_id}. "
        f"Amount: {sold_amount:.2f}, Bet ID: {receipt.bet_id or 'unknown'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
