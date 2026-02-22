CREATE TABLE IF NOT EXISTS equity_snapshots (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    snapshot_type TEXT NOT NULL,
    cash_balance NUMERIC,
    positions_value NUMERIC,
    bankroll NUMERIC,
    gross_exposure NUMERIC,
    open_positions INTEGER,
    snapshot_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_run ON equity_snapshots(run_id);
