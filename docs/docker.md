## Docker Setup (Separate Agent Containers)

This setup runs a shared market fetcher, three isolated agent containers, and a Postgres container.

### Required env vars
- `OPENAI_API_KEY`
- `CLAUDE_API_KEY`
- `GEMINI_API_KEY`
- `MANIFOLD_API_KEY_OPENAI`
- `MANIFOLD_API_KEY_CLAUDE`
- `MANIFOLD_API_KEY_GEMINI`

Optional:
- `AGENT_MAX_ATTEMPTS` (default `2`)
- `AGENT_MARKET_CACHE_LIMIT` (default `25`)
- `MARKET_CACHE_WAIT_SECONDS` (default `120`)
- `DATABASE_URL_DOCKER` (override the default Postgres URL used inside containers)
 
Schema:
- The Postgres container initializes the schema from `db/init/*.sql`.
- Agents set `AGENT_SKIP_SCHEMA_INIT=1` to avoid concurrent schema creation.

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
