## Docker Setup (Separate Agent Containers)

This setup runs a shared market fetcher, three isolated agent containers, and a Postgres container.

### Required env vars
- `OPENAI_API_KEY`
- `CLAUDE_API_KEY`
- `GEMINI_API_KEY`
- `MANIFOLD_API_KEY_OPENAI`
- `MANIFOLD_API_KEY_CLAUDE`
- `MANIFOLD_API_KEY_GEMINI`

Required:
- `DATABASE_URL` (Postgres URL used inside containers; use host `postgres`)

Optional:
- `AGENT_MAX_ATTEMPTS` (default `2`)
- `AGENT_MARKET_CACHE_LIMIT` (default `25`)
- `MARKET_CACHE_WAIT_SECONDS` (default `120`)
 
Schema:
- The Postgres container initializes the schema from `db/init/*.sql`.
- Agents do not create schema at runtime.
- Agents are seeded from `agents.json` during first-time Postgres initialization.
- The market fetcher creates a session row and run placeholders for each agent.

Notes:
- The runtime will backfill `ANTHROPIC_API_KEY` from `CLAUDE_API_KEY` and `GOOGLE_API_KEY` from `GEMINI_API_KEY` if needed.

### Run
```bash
docker compose up --build
```

### Notes
- Market cache is written to `data/shared_markets.json`.
- Each agent writes JSONL output to `results/`.
- Postgres data is stored in the `pg_data` volume.
