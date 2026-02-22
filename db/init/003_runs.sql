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
    tokens_in INTEGER,
    tokens_out INTEGER,
    tokens_total INTEGER,
    cash_netted NUMERIC,
    current_balance NUMERIC,
    cash_balance NUMERIC,
    position_balance NUMERIC,
    bankroll NUMERIC,
    plan_output_json JSONB,
    execution_output TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
