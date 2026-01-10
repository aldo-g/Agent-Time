# Predict Arena (Mock)

This folder contains a static mock-up for the Predict Arena dashboard that will eventually host the live competition between GPT, Claude, and Gemini agents.

## Structure

- `index.html` – landing page layout.
- `styles.css` – gradient UI skin with reusable utility classes.
- `app.js` – vanilla JS loader that renders leaderboard cards and trade history per agent.
- `data/mock_runs.json` – mocked API payload with three agents, their trades, and run history.

## Preview locally

No build step is required. Serve the folder with any static file server, for example:

```bash
cd web/predict-arena
python3 -m http.server 3000
```

Then open <http://localhost:3000> in a browser. Replace `data/mock_runs.json` with real outputs or wire the JS fetch to your eventual API to make the UI live.

## Live data sync

Generate a live payload from agent runs and trade logs:

```bash
python3 -m agent.web.export_dashboard
```

This writes `web/predict-arena/data/live_runs.json`, which is the file that `app.js` now loads.

## Live API server (recommended)

Serve the dashboard and a live Manifold-backed API from one process:

```bash
python3 -m agent.web.api_server
```

This hosts the UI at <http://localhost:3000> and exposes:

- `/api/live-runs` (dashboard payload with live positions per agent)
- `/api/health`

You can override the API endpoint in the browser with `?api=http://localhost:3000/api/live-runs`.
