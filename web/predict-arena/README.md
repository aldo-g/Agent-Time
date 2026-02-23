# Predict Arena (Mock)

This folder contains a static mock-up for the Predict Arena dashboard for a single ChatGPT trading agent.

## Structure

- `index.html` – landing page layout.
- `styles.css` – gradient UI skin with reusable utility classes.
- `app.js` – vanilla JS loader that renders leaderboard cards and trade history per agent.
- `data/mock_runs.json` – mocked API payload with one agent, trade examples, and run history.

## Preview locally

No build step is required. Serve the folder with any static file server, for example:

```bash
cd web/predict-arena
python3 -m http.server 3000
```

Then open <http://localhost:3000> in a browser.

## Live data sync

Generate a live payload. If `DATABASE_URL` is set, it reads from Postgres (`runs`, `trade_executions`, `open_positions`, `equity_snapshots`); otherwise it falls back to JSONL logs.

```bash
python3 -m agent.web.export_dashboard
```

This writes `web/predict-arena/data/live_runs.json` for offline inspection/export.

## Live API server (recommended)

Serve the dashboard and a live Manifold-backed API from one process:

```bash
python3 -m agent.web.api_server
```

This hosts the UI at <http://localhost:3000> and exposes:

- `/api/live-runs` (DB-backed payload by default)
- `/api/health`

Use `?live=1` to additionally hydrate positions directly from Manifold for that request.

You can override the API endpoint in the browser with `?api=http://localhost:3000/api/live-runs`.

The dashboard now loads data from `/api/live-runs` only (no file fallback), so stale `data/live_runs.json` files cannot be displayed by mistake.
