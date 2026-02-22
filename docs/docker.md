## Docker Setup (Single ChatGPT Agent Container)

This setup runs a shared market fetcher, one isolated ChatGPT agent container, and a Postgres container.

### Required env vars
- `OPENAI_API_KEY`
- `MANIFOLD_API_KEY_OPENAI`

Required:
- `DATABASE_URL` (Postgres URL used inside containers; use host `postgres`)

Optional:
- `AGENT_MAX_ATTEMPTS` (default `2`)
- `AGENT_MARKET_CACHE_LIMIT` (default `25`)
- `MARKET_CACHE_WAIT_SECONDS` (default `120`)
 
Schema:
- The Postgres container initializes the schema from `db/init/*.sql`.
- Runtime components also call `ensure_schema()` for compatibility with existing volumes.
- The market fetcher reads `agent.json`, creates a session row, and inserts one run placeholder.

### Run
```bash
docker compose up --build
```

### Notes
- Market cache is written to `data/shared_markets.json`.
- The agent writes JSONL output to `results/gpt/`.
- Postgres data is stored in the `pg_data` volume.
