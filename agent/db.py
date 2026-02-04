"""Postgres persistence for Agent-Time runs and trades."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import utils.env_loader as env_loader  # noqa: F401


def _require_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Add it to .env.")
    return url


@dataclass
class TradeRecord:
    agent_name: str
    trade_text: str
    reason: str | None
    amount: float | None
    status: str
    market_id: str | None = None
    market_slug: str | None = None
    decision_confidence: float | None = None
    edge: float | None = None
    error: str | None = None
    created_at: datetime | None = None


class DbWriter:
    """Lightweight Postgres writer using psycopg (v3)."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "psycopg is not installed. Install it with `pip install psycopg[binary]`."
            ) from exc
        self._psycopg = psycopg
        self._errors = psycopg.errors
        self._dsn = dsn

    @classmethod
    def from_env(cls) -> "DbWriter":
        return cls(_require_database_url())

    def connect(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id BIGSERIAL PRIMARY KEY,
                        started_at TIMESTAMPTZ NOT NULL,
                        finished_at TIMESTAMPTZ,
                        market_cache_path TEXT,
                        market_count INTEGER,
                        notes TEXT
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agents (
                        id BIGSERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        model_provider TEXT,
                        model TEXT,
                        current_balance NUMERIC,
                        cash_balance NUMERIC,
                        position_balance NUMERIC,
                        last_seen_at TIMESTAMPTZ
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id BIGSERIAL PRIMARY KEY,
                        session_id BIGINT REFERENCES sessions(id) ON DELETE CASCADE,
                        agent_id BIGINT REFERENCES agents(id) ON DELETE CASCADE,
                        started_at TIMESTAMPTZ NOT NULL,
                        finished_at TIMESTAMPTZ,
                        run_duration_ms INTEGER,
                        success BOOLEAN NOT NULL,
                        error TEXT,
                        no_trade_reason TEXT,
                        tool_calls_count INTEGER,
                        tokens_in INTEGER,
                        tokens_out INTEGER,
                        tokens_total INTEGER,
                        cash_netted NUMERIC,
                        bankroll NUMERIC
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
                        agent_id BIGINT REFERENCES agents(id) ON DELETE CASCADE,
                        market_id TEXT,
                        market_slug TEXT,
                        trade_text TEXT NOT NULL,
                        reason TEXT,
                        amount NUMERIC,
                        status TEXT NOT NULL,
                        error TEXT,
                        decision_confidence NUMERIC,
                        edge NUMERIC,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS agent_id BIGINT;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS session_id BIGINT;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_duration_ms INTEGER;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tool_calls_count INTEGER;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tokens_in INTEGER;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tokens_out INTEGER;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tokens_total INTEGER;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS cash_netted NUMERIC;"
                )
                cur.execute(
                    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS bankroll NUMERIC;"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS agent_id BIGINT;"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_id TEXT;"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_slug TEXT;"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS decision_confidence NUMERIC;"
                )
                cur.execute(
                    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS edge NUMERIC;"
                )
                cur.execute(
                    "ALTER TABLE agents DROP COLUMN IF EXISTS total_token_cost;"
                )
                cur.execute(
                    "ALTER TABLE runs DROP COLUMN IF EXISTS model_price;"
                )
                cur.execute(
                    "ALTER TABLE runs DROP COLUMN IF EXISTS total_token_cost;"
                )
                cur.execute(
                    "ALTER TABLE runs DROP COLUMN IF EXISTS cash_balance;"
                )
                cur.execute(
                    "ALTER TABLE runs DROP COLUMN IF EXISTS positions_value;"
                )
                try:
                    cur.execute("ALTER TABLE runs ALTER COLUMN agent_name DROP NOT NULL;")
                except self._errors.UndefinedColumn:
                    pass
                try:
                    cur.execute("ALTER TABLE trades ALTER COLUMN agent_name DROP NOT NULL;")
                except self._errors.UndefinedColumn:
                    pass

    def create_session(
        self,
        *,
        started_at: datetime,
        market_cache_path: Optional[str],
        market_count: Optional[int],
        notes: Optional[str] = None,
    ) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (
                        started_at,
                        market_cache_path,
                        market_count,
                        notes
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        started_at,
                        market_cache_path,
                        market_count,
                        notes,
                    ),
                )
                session_id = cur.fetchone()[0]
        return int(session_id)

    def finish_session(self, *, session_id: int, finished_at: datetime) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET finished_at = %s WHERE id = %s;",
                    (finished_at, session_id),
                )

    def upsert_agent(
        self,
        *,
        agent_name: str,
        model_provider: str,
        model: str,
        current_balance: Optional[float],
        cash_balance: Optional[float],
        position_balance: Optional[float],
        last_seen_at: datetime,
    ) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agents (
                        name,
                        model_provider,
                        model,
                        current_balance,
                        cash_balance,
                        position_balance,
                        last_seen_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name)
                    DO UPDATE SET
                        model_provider = EXCLUDED.model_provider,
                        model = EXCLUDED.model,
                        current_balance = EXCLUDED.current_balance,
                        cash_balance = EXCLUDED.cash_balance,
                        position_balance = EXCLUDED.position_balance,
                        last_seen_at = EXCLUDED.last_seen_at
                    RETURNING id;
                    """,
                    (
                        agent_name,
                        model_provider,
                        model,
                        current_balance,
                        cash_balance,
                        position_balance,
                        last_seen_at,
                    ),
                )
                agent_id = cur.fetchone()[0]
        return int(agent_id)

    def insert_run(
        self,
        *,
        session_id: int,
        agent_id: int,
        started_at: datetime,
        finished_at: Optional[datetime],
        run_duration_ms: Optional[int],
        success: bool,
        error: Optional[str],
        no_trade_reason: Optional[str],
        tool_calls_count: Optional[int],
        tokens_in: Optional[int],
        tokens_out: Optional[int],
        tokens_total: Optional[int],
        cash_netted: Optional[float],
        bankroll: Optional[float],
    ) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (
                        session_id,
                        agent_id,
                        started_at,
                        finished_at,
                        run_duration_ms,
                        success,
                        error,
                        no_trade_reason,
                        tool_calls_count,
                        tokens_in,
                        tokens_out,
                        tokens_total,
                        cash_netted,
                        bankroll
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        session_id,
                        agent_id,
                        started_at,
                        finished_at,
                        run_duration_ms,
                        success,
                        error,
                        no_trade_reason,
                        tool_calls_count,
                        tokens_in,
                        tokens_out,
                        tokens_total,
                        cash_netted,
                        bankroll,
                    ),
                )
                run_id = cur.fetchone()[0]
        return int(run_id)

    def insert_trades(self, *, run_id: int, agent_id: int, trades: Iterable[TradeRecord]) -> None:
        rows = [
            (
                run_id,
                agent_id,
                trade.market_id,
                trade.market_slug,
                trade.trade_text,
                trade.reason,
                trade.amount,
                trade.status,
                trade.error,
                trade.decision_confidence,
                trade.edge,
                trade.created_at or datetime.now(timezone.utc),
            )
            for trade in trades
        ]
        if not rows:
            return
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO trades (
                        run_id,
                        agent_id,
                        market_id,
                        market_slug,
                        trade_text,
                        reason,
                        amount,
                        status,
                        error,
                        decision_confidence,
                        edge,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    rows,
                )
