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

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_run ON equity_snapshots(run_id);

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

CREATE INDEX IF NOT EXISTS idx_open_positions_run ON open_positions(run_id);
CREATE INDEX IF NOT EXISTS idx_open_positions_market ON open_positions(market_id);
