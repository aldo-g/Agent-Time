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

CREATE INDEX IF NOT EXISTS idx_run_actions_run ON run_actions(run_id);
CREATE INDEX IF NOT EXISTS idx_run_actions_market ON run_actions(market_id);
CREATE INDEX IF NOT EXISTS idx_trade_exec_run ON trade_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_trade_exec_action ON trade_executions(run_action_id);
CREATE INDEX IF NOT EXISTS idx_trade_exec_bet_id ON trade_executions(bet_id);
CREATE INDEX IF NOT EXISTS idx_trade_exec_created_at ON trade_executions(created_at);
