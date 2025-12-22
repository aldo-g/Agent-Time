"""Helpers for fetching Manifold market details and placing bets."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import utils.env_loader as env_loader  # noqa: F401
from agent.manifold.constants import MANIFOLD_API_ROOT, RESOLUTION_CUTOFF_MS

USER_AGENT = "AgentTimeBot/1.0 (+https://manifold.markets)"


@dataclass
class OutcomeOption:
    """Available outcome option for a Manifold market."""

    outcome: str
    label: str
    probability: Optional[float]
    answer_id: Optional[str] = None


@dataclass
class MarketDetails:
    """Summary metadata for a Manifold market."""

    market_id: str
    slug: str | None
    url: str | None
    question: str
    outcome_type: str
    answers: List[OutcomeOption]
    close_time: Optional[int]
    raw: Dict[str, object]


@dataclass
class BetReceipt:
    """Summary of a submitted bet."""

    bet_id: Optional[str]
    outcome: str
    amount: float
    shares: Optional[float]
    probability: Optional[float]
    response: object


def fetch_market_details(identifier: str) -> MarketDetails:
    """Return structured metadata for the given Manifold market id or slug."""
    payload = _fetch_market_payload(identifier)
    market_id = str(payload.get("id") or payload.get("_id") or identifier)
    slug = payload.get("slug")
    url = payload.get("url")
    question = payload.get("question") or payload.get("title") or "Untitled market"
    outcome_type = str(payload.get("outcomeType") or payload.get("mechanism") or "").upper()
    answers: List[OutcomeOption] = []
    probability = payload.get("probability")
    if outcome_type in {"BINARY", "PSEUDO_NUMERIC"}:
        try:
            prob = float(probability)
        except (TypeError, ValueError):
            prob = None
        answers.append(OutcomeOption(outcome="YES", label="YES", probability=prob))
        answers.append(
            OutcomeOption(outcome="NO", label="NO", probability=(1 - prob) if prob is not None else None)
        )
    else:
        for answer in payload.get("answers") or []:
            if not isinstance(answer, dict):
                continue
            label = answer.get("text") or answer.get("name") or f"Answer {answer.get('index', '-')}"
            try:
                prob = float(answer.get("probability"))
            except (TypeError, ValueError):
                prob = None
            answers.append(
                OutcomeOption(
                    outcome=str(answer.get("outcome") or answer.get("id") or label),
                    label=str(label),
                    probability=prob,
                    answer_id=answer.get("id"),
                )
            )
    close_time = payload.get("closeTime")
    try:
        close_time_ms = int(close_time) if close_time is not None else None
    except (TypeError, ValueError):
        close_time_ms = None
    return MarketDetails(
        market_id=market_id,
        slug=str(slug) if slug else None,
        url=str(url) if url else None,
        question=str(question),
        outcome_type=outcome_type,
        answers=answers,
        close_time=close_time_ms,
        raw=payload,
    )


def place_bet(
    *,
    market_id: str,
    outcome: str,
    amount: float,
    limit_prob: Optional[float] = None,
    answer_id: Optional[str] = None,
) -> BetReceipt:
    """Submit a Manifold bet using play-money Mana."""
    if amount <= 0:
        raise ValueError("amount must be positive.")
    resolved_id = fetch_market_details(market_id).market_id
    body: Dict[str, object] = {
        "amount": amount,
        "contractId": resolved_id,
        "outcome": outcome.upper(),
    }
    if answer_id:
        body["answerId"] = answer_id
    if limit_prob is not None:
        if not 0.0 < limit_prob < 1.0:
            raise ValueError("limit_prob must be between 0 and 1.")
        body["limitProb"] = limit_prob
    payload = _api_request("/bet", method="POST", body=body)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected response from Manifold bet endpoint.")
    bet_id = payload.get("betId") or payload.get("id")
    shares = payload.get("shares")
    try:
        shares_value = float(shares) if shares is not None else None
    except (TypeError, ValueError):
        shares_value = None
    try:
        amount_value = float(payload.get("amount"))
    except (TypeError, ValueError):
        amount_value = amount
    try:
        prob_value = float(payload.get("probAfter"))
    except (TypeError, ValueError):
        prob_value = None
    return BetReceipt(
        bet_id=bet_id if isinstance(bet_id, str) else None,
        outcome=outcome.upper(),
        amount=amount_value,
        shares=shares_value,
        probability=prob_value,
        response=payload,
    )


def lookup_answer_id(details: MarketDetails, label: str) -> Optional[str]:
    """Best-effort match from a user-provided label to a Manifold answer id."""
    if not label:
        return None
    desired = label.strip().lower()
    for answer in details.answers:
        if answer.answer_id and answer.label.strip().lower() == desired:
            return answer.answer_id
    return None


def _fetch_market_payload(identifier: str) -> Dict[str, object]:
    if not identifier:
        raise RuntimeError("market identifier is required.")
    candidates = [
        f"/market/{urllib.parse.quote(identifier, safe='')}",
        f"/slug/{urllib.parse.quote(identifier, safe='')}",
    ]
    last_error: Optional[Exception] = None
    for path in candidates:
        try:
            payload = _api_request(path)
        except urllib.error.HTTPError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict) and payload.get("id"):
            return payload
    if last_error:
        raise RuntimeError(f"Unable to load Manifold market {identifier}: {last_error}") from last_error
    raise RuntimeError(f"Unable to load Manifold market {identifier}: not found.")


def _api_request(path: str, *, method: str = "GET", body: object | None = None) -> object:
    url = _build_url(path)
    headers = _auth_headers()
    data_bytes = None
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data_bytes = bytes(body)
        else:
            data_text = json.dumps(body)
            data_bytes = data_text.encode("utf-8")
    request = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(
                    url=url,
                    code=response.status,
                    msg=response.reason,
                    hdrs=response.headers,
                    fp=response,
                )
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = _read_error_body(exc)
        raise RuntimeError(f"Manifold API request failed ({exc.code} {exc.reason}): {detail}") from exc


def _build_url(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{MANIFOLD_API_ROOT}{normalized}"


def _auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("MANIFOLD_API_KEY")
    if not api_key:
        raise RuntimeError("Set MANIFOLD_API_KEY to access authenticated Manifold endpoints.")
    return {
        "Authorization": f"Key {api_key}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body or "no response body"


__all__ = [
    "BetReceipt",
    "MarketDetails",
    "OutcomeOption",
    "fetch_market_details",
    "lookup_answer_id",
    "place_bet",
]
