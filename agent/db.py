"""Postgres persistence for Agent-Time runs and trade lifecycle records."""

from __future__ import annotations

import json
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


def _json_or_none(payload: object | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload)


@dataclass
class RunActionRecord:
    action_index: int
    action_type: str
    market_id: str | None = None
    outcome: str | None = None
    amount: float | None = None
    shares: float | None = None
    belief_prob: float | None = None
    market_prob: float | None = None
    edge_at_plan: float | None = None
    limit_prob: float | None = None
    answer: str | None = None
    requires_news_catalyst: bool | None = None
    catalyst_urls: list[str] | None = None
    reason: str | None = None
    status: str = "planned"
    skip_reason: str | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None


@dataclass
class TradeExecutionRecord:
    run_action_id: int | None
    market_id: str | None
    market_slug: str | None
    action: str | None
    outcome: str | None
    amount: float | None
    shares: float | None
    prob_before: float | None = None
    prob_after: float | None = None
    bet_id: str | None = None
    status: str = "executed"
    error: str | None = None
    reason: str | None = None
    summary: str | None = None
    raw_response: dict | None = None
    created_at: datetime | None = None


@dataclass
class EquitySnapshotRecord:
    cash_balance: float | None = None
    positions_value: float | None = None
    bankroll: float | None = None
    gross_exposure: float | None = None
    open_positions: int | None = None
    snapshot_json: dict | None = None
    created_at: datetime | None = None


@dataclass
class OpenPositionRecord:
    market_id: str
    market_slug: str | None = None
    question: str | None = None
    outcome: str | None = None
    shares: float | None = None
    avg_price: float | None = None
    mark_price: float | None = None
    position_value: float | None = None
    pnl: float | None = None
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

    def ping(self) -> None:
        """Verify the database is reachable before doing any writes."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")

    def ensure_schema(self) -> None:
        """Create or migrate the single-agent run/action/execution schema."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id BIGSERIAL PRIMARY KEY,
                        started_at TIMESTAMPTZ NOT NULL,
                        finished_at TIMESTAMPTZ,
                        market_json JSONB,
                        notes TEXT
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        id BIGSERIAL PRIMARY KEY,
                        session_id BIGINT REFERENCES sessions(id) ON DELETE CASCADE,
                        model_provider TEXT,
                        model TEXT,
                        started_at TIMESTAMPTZ NOT NULL,
                        finished_at TIMESTAMPTZ,
                        run_duration_ms INTEGER,
                        success BOOLEAN,
                        error TEXT,
                        no_trade_reason TEXT,
                        tool_calls_count INTEGER,
                        cash_netted NUMERIC,
                        current_balance NUMERIC,
                        cash_balance NUMERIC,
                        position_balance NUMERIC,
                        bankroll NUMERIC,
                        plan_output_json JSONB,
                        execution_output TEXT,
                        metadata JSONB
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_actions (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        action_index INTEGER NOT NULL,
                        action_type TEXT NOT NULL,
                        market_id TEXT,
                        outcome TEXT,
                        amount NUMERIC,
                        shares NUMERIC,
                        belief_prob NUMERIC,
                        market_prob NUMERIC,
                        edge_at_plan NUMERIC,
                        limit_prob NUMERIC,
                        answer TEXT,
                        requires_news_catalyst BOOLEAN,
                        catalyst_urls JSONB,
                        reason TEXT,
                        status TEXT NOT NULL,
                        skip_reason TEXT,
                        failure_reason TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trade_executions (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        run_action_id BIGINT REFERENCES run_actions(id) ON DELETE SET NULL,
                        market_id TEXT,
                        market_slug TEXT,
                        action TEXT,
                        outcome TEXT,
                        amount NUMERIC,
                        shares NUMERIC,
                        prob_before NUMERIC,
                        prob_after NUMERIC,
                        bet_id TEXT,
                        status TEXT NOT NULL,
                        error TEXT,
                        reason TEXT,
                        summary TEXT,
                        raw_response JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS equity_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        snapshot_type TEXT NOT NULL DEFAULT 'post',
                        cash_balance NUMERIC,
                        positions_value NUMERIC,
                        bankroll NUMERIC,
                        gross_exposure NUMERIC,
                        open_positions INTEGER,
                        snapshot_json JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS open_positions (
                        id BIGSERIAL PRIMARY KEY,
                        run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        market_id TEXT NOT NULL,
                        market_slug TEXT,
                        question TEXT,
                        outcome TEXT,
                        shares NUMERIC,
                        avg_price NUMERIC,
                        mark_price NUMERIC,
                        position_value NUMERIC,
                        pnl NUMERIC,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )

                # Legacy-table compatibility: keep old tables if present and migrate runs to single-agent mode.
                cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS model_provider TEXT;")
                cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS model TEXT;")
                cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS plan_output_json JSONB;")
                cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS execution_output TEXT;")
                cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS metadata JSONB;")
                cur.execute("ALTER TABLE runs DROP COLUMN IF EXISTS tokens_in;")
                cur.execute("ALTER TABLE runs DROP COLUMN IF EXISTS tokens_out;")
                cur.execute("ALTER TABLE runs DROP COLUMN IF EXISTS tokens_total;")
                # Older schemas had runs.agent_id as NOT NULL; make it optional so placeholders can be created
                # without a separate agents table.
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                              AND table_name = 'runs'
                              AND column_name = 'agent_id'
                        ) THEN
                            EXECUTE 'ALTER TABLE runs ALTER COLUMN agent_id DROP NOT NULL';
                        END IF;
                    EXCEPTION WHEN undefined_table OR undefined_column THEN
                        NULL;
                    END
                    $$;
                    """
                )

                cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_run_actions_run ON run_actions(run_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_run_actions_market ON run_actions(market_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_exec_run ON trade_executions(run_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_exec_action ON trade_executions(run_action_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_exec_bet_id ON trade_executions(bet_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_exec_created_at ON trade_executions(created_at);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_equity_snapshots_run ON equity_snapshots(run_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_open_positions_run ON open_positions(run_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_open_positions_market ON open_positions(market_id);")
                cur.execute(
                    """
                    ALTER TABLE equity_snapshots
                    ALTER COLUMN snapshot_type SET DEFAULT 'post';
                    """
                )

    def create_session(
        self,
        *,
        started_at: datetime,
        market_json: Optional[dict],
        notes: Optional[str] = None,
    ) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (started_at, market_json, notes)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        started_at,
                        _json_or_none(market_json),
                        notes,
                    ),
                )
                session_id = cur.fetchone()[0]
        return int(session_id)

    def get_latest_session(self) -> Optional[dict]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, started_at
                    FROM sessions
                    ORDER BY started_at DESC
                    LIMIT 1;
                    """
                )
                row = cur.fetchone()
                if not row:
                    return None
                session_id, started_at = row
        return {"id": int(session_id), "started_at": started_at}

    def get_run_id(self, *, session_id: int) -> Optional[int]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM runs
                    WHERE session_id = %s
                    ORDER BY started_at DESC, id DESC
                    LIMIT 1;
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return int(row[0])

    def finish_session(self, *, session_id: int, finished_at: datetime) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET finished_at = %s WHERE id = %s;",
                    (finished_at, session_id),
                )

    def create_run_placeholder(
        self,
        *,
        session_id: int,
        model_provider: str | None,
        model: str | None,
        started_at: datetime,
    ) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (
                        session_id,
                        model_provider,
                        model,
                        started_at,
                        success
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (session_id, model_provider, model, started_at, None),
                )
                run_id = cur.fetchone()[0]
        return int(run_id)

    def update_run(
        self,
        *,
        run_id: int,
        started_at: datetime,
        finished_at: Optional[datetime],
        run_duration_ms: Optional[int],
        success: Optional[bool],
        error: Optional[str],
        no_trade_reason: Optional[str],
        tool_calls_count: Optional[int],
        cash_netted: Optional[float],
        current_balance: Optional[float],
        cash_balance: Optional[float],
        position_balance: Optional[float],
        bankroll: Optional[float],
        plan_output_json: Optional[dict] = None,
        execution_output: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE runs
                    SET
                        started_at = %s,
                        finished_at = %s,
                        run_duration_ms = %s,
                        success = %s,
                        error = %s,
                        no_trade_reason = %s,
                        tool_calls_count = %s,
                        cash_netted = %s,
                        current_balance = %s,
                        cash_balance = %s,
                        position_balance = %s,
                        bankroll = %s,
                        plan_output_json = %s,
                        execution_output = %s,
                        metadata = %s
                    WHERE id = %s;
                    """,
                    (
                        started_at,
                        finished_at,
                        run_duration_ms,
                        success,
                        error,
                        no_trade_reason,
                        tool_calls_count,
                        cash_netted,
                        current_balance,
                        cash_balance,
                        position_balance,
                        bankroll,
                        _json_or_none(plan_output_json),
                        execution_output,
                        _json_or_none(metadata),
                        run_id,
                    ),
                )

    def insert_run_actions(self, *, run_id: int, actions: Iterable[RunActionRecord]) -> list[int]:
        action_ids: list[int] = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                for action in actions:
                    cur.execute(
                        """
                        INSERT INTO run_actions (
                            run_id,
                            action_index,
                            action_type,
                            market_id,
                            outcome,
                            amount,
                            shares,
                            belief_prob,
                            market_prob,
                            edge_at_plan,
                            limit_prob,
                            answer,
                            requires_news_catalyst,
                            catalyst_urls,
                            reason,
                            status,
                            skip_reason,
                            failure_reason,
                            created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            run_id,
                            action.action_index,
                            action.action_type,
                            action.market_id,
                            action.outcome,
                            action.amount,
                            action.shares,
                            action.belief_prob,
                            action.market_prob,
                            action.edge_at_plan,
                            action.limit_prob,
                            action.answer,
                            action.requires_news_catalyst,
                            _json_or_none(action.catalyst_urls),
                            action.reason,
                            action.status,
                            action.skip_reason,
                            action.failure_reason,
                            action.created_at or datetime.now(timezone.utc),
                        ),
                    )
                    action_ids.append(int(cur.fetchone()[0]))
        return action_ids

    def insert_trade_executions(self, *, run_id: int, executions: Iterable[TradeExecutionRecord]) -> None:
        rows = [
            (
                run_id,
                execution.run_action_id,
                execution.market_id,
                execution.market_slug,
                execution.action,
                execution.outcome,
                execution.amount,
                execution.shares,
                execution.prob_before,
                execution.prob_after,
                execution.bet_id,
                execution.status,
                execution.error,
                execution.reason,
                execution.summary,
                _json_or_none(execution.raw_response),
                execution.created_at or datetime.now(timezone.utc),
            )
            for execution in executions
        ]
        if not rows:
            return
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO trade_executions (
                        run_id,
                        run_action_id,
                        market_id,
                        market_slug,
                        action,
                        outcome,
                        amount,
                        shares,
                        prob_before,
                        prob_after,
                        bet_id,
                        status,
                        error,
                        reason,
                        summary,
                        raw_response,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    rows,
                )

    def insert_equity_snapshots(self, *, run_id: int, snapshots: Iterable[EquitySnapshotRecord]) -> None:
        rows = [
            (
                run_id,
                "post",
                snapshot.cash_balance,
                snapshot.positions_value,
                snapshot.bankroll,
                snapshot.gross_exposure,
                snapshot.open_positions,
                _json_or_none(snapshot.snapshot_json),
                snapshot.created_at or datetime.now(timezone.utc),
            )
            for snapshot in snapshots
        ]
        if not rows:
            return
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM equity_snapshots WHERE run_id = %s;", (run_id,))
                cur.executemany(
                    """
                    INSERT INTO equity_snapshots (
                        run_id,
                        snapshot_type,
                        cash_balance,
                        positions_value,
                        bankroll,
                        gross_exposure,
                        open_positions,
                        snapshot_json,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    rows,
                )

    def insert_open_positions(self, *, run_id: int, positions: Iterable[OpenPositionRecord]) -> None:
        rows = [
            (
                run_id,
                position.market_id,
                position.market_slug,
                position.question,
                position.outcome,
                position.shares,
                position.avg_price,
                position.mark_price,
                position.position_value,
                position.pnl,
                position.created_at or datetime.now(timezone.utc),
            )
            for position in positions
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM open_positions WHERE run_id = %s;", (run_id,))
                if not rows:
                    return
                cur.executemany(
                    """
                    INSERT INTO open_positions (
                        run_id,
                        market_id,
                        market_slug,
                        question,
                        outcome,
                        shares,
                        avg_price,
                        mark_price,
                        position_value,
                        pnl,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    rows,
                )
