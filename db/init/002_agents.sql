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

CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
