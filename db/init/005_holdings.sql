CREATE TABLE IF NOT EXISTS holdings (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT REFERENCES agents(id) ON DELETE CASCADE,
    market_id TEXT NOT NULL,
    market_slug TEXT,
    outcome TEXT,
    shares NUMERIC,
    status TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_id, market_id, outcome)
);

CREATE INDEX IF NOT EXISTS idx_holdings_agent ON holdings(agent_id);
CREATE INDEX IF NOT EXISTS idx_holdings_market ON holdings(market_id);
