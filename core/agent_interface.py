"""Core agent interface definitions for the Agent-Time project."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Union


class OutcomeSide(str, Enum):
    """Canonical Manifold-style outcomes; more can be added later."""

    YES = "YES"
    NO = "NO"
    BINARY = "BINARY"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"


@dataclass(frozen=True)
class MarketObservation:
    market_id: str
    question: str
    close_time: datetime
    last_updated: datetime
    probability: float
    volume: float
    liquidity: float
    market_type: str
    additional_metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSnapshot:
    market_id: str
    outcome: str
    shares: float
    average_price: float
    mark_price: float

    @property
    def unrealized_pnl(self) -> float:
        return (self.mark_price - self.average_price) * self.shares


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash_balance: float
    total_exposure: float
    positions: Sequence[PositionSnapshot] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentObservation:
    """What the agent can "see" before making a decision."""

    timestamp: datetime
    market: MarketObservation
    portfolio: PortfolioSnapshot


@dataclass(frozen=True)
class RewardSignal:
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float


@dataclass(frozen=True)
class RiskConstraintResult:
    name: str
    passed: bool
    limit_value: float
    observed_value: float
    message: Optional[str] = None


@dataclass(frozen=True)
class EvidenceReference:
    url: str
    description: Optional[str] = None


@dataclass(frozen=True)
class BeliefEstimate:
    probability: float
    confidence: float


@dataclass(frozen=True)
class BetSizing:
    stake: float
    max_allowed: float
    rationale: Optional[str] = None
    inputs: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentActionBase:
    market_id: str


@dataclass(frozen=True)
class PlaceBetAction(AgentActionBase):
    outcome: OutcomeSide
    amount: float
    limit_probability: Optional[float] = None


@dataclass(frozen=True)
class SellPositionAction(AgentActionBase):
    shares: Optional[float] = None
    amount: Optional[float] = None


@dataclass(frozen=True)
class NoOpAction(AgentActionBase):
    reason: Optional[str] = None


AgentAction = Union[PlaceBetAction, SellPositionAction, NoOpAction]


@dataclass(frozen=True)
class DecisionPacket:
    market_id: str
    question: str
    close_time: datetime
    market_probability: float
    belief: BeliefEstimate
    expected_value: float
    bet_sizing: BetSizing
    action: AgentAction
    risk_results: Sequence[RiskConstraintResult] = field(default_factory=tuple)
    rationale: str = ""
    evidence: Sequence[EvidenceReference] = field(default_factory=tuple)


class Agent(Protocol):
    """Standard interface every agent implementation must satisfy."""

    def decide(self, observation: AgentObservation) -> DecisionPacket:
        ...

    def reward(self, signal: RewardSignal) -> None:
        """Optional hook for updating strategy state from PnL."""


class DecisionLogger(Protocol):
    def log(self, packet: DecisionPacket) -> None:
        ...


class RiskEngine(Protocol):
    def evaluate(self, observation: AgentObservation, action: AgentAction) -> Sequence[RiskConstraintResult]:
        ...
