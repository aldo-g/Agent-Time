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

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
