-- Seed agents from agents.json on first database initialization.
-- The file is mounted into /agents.json by docker-compose.
\set agents_json `cat /agents.json | tr -d '\n\r'`

INSERT INTO agents (name, model_provider, model, last_seen_at)
SELECT
    agent->>'name' AS name,
    agent->>'model_provider' AS model_provider,
    agent->>'model' AS model,
    NULL AS last_seen_at
FROM jsonb_array_elements(:'agents_json'::jsonb) AS agent
ON CONFLICT (name) DO UPDATE
SET
    model_provider = EXCLUDED.model_provider,
    model = EXCLUDED.model;
