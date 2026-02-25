# Agent-Time

Autonomous prediction-market trading system for Manifold, with scheduled execution, portfolio tracking, and a live web dashboard.

## What This Project Does

- Runs an LLM trading agent on a daily schedule.
- Fetches markets, evaluates opportunities, and executes trades.
- Stores run + trade history in Postgres (optional but recommended).
- Serves a dashboard frontend/API for performance and trade visibility.
- Deploys to AWS (EC2-first path) with Terraform + Docker.

## Stack

- `Python 3.11` agent runtime
- `LangChain + OpenAI` agent orchestration
- `Postgres` persistence (`psycopg`)
- `Vanilla JS + static UI` dashboard (`web/predict-arena`)
- `Terraform + EC2 + ECR + SSM + CloudWatch + ALB + ACM + Route53` AWS infrastructure

## Architecture

1. `agent.fetch_markets` builds market context.
2. `agent.single_runner` runs the configured agent and can place trades.
3. Results/trades are written to JSONL and optionally to Postgres.
4. `agent.web.api_server` serves the dashboard and live API.
5. AWS systemd timers run the bot daily.

## Local Run (Docker)

```bash
cp .env.example .env
docker compose up --build
```

In a second terminal, run the dashboard/API server:

```bash
python3 -m agent.web.api_server --host 127.0.0.1 --port 3000
```

Then open:

- Dashboard/API: `http://localhost:3000`
- Postgres: `localhost:5432`

Local Docker details are in [docs/docker.md](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/docs/docker.md).

## Production AWS (EC2-First, Publishable Dashboard)

Use the production helper script:

```bash
./scripts/deploy_aws_prod.sh
```

For HTTPS + custom domain:

```bash
TFVARS_TEMPLATE=infra/terraform/aws-bots/terraform.tfvars.prod.https.example ./scripts/deploy_aws_prod.sh
```

Replace domain placeholders in `infra/terraform/aws-bots/terraform.tfvars` before first HTTPS deploy.

This script:

1. Initializes/applies Terraform in `infra/terraform/aws-bots`.
2. Builds and pushes your Docker image to ECR.
3. Restarts bot/dashboard services via SSM.
4. Prints public dashboard URL(s).

If you need to view URLs again:

```bash
./scripts/dashboard_url.sh
```

Infra details and manual commands are documented in [infra/terraform/aws-bots/README.md](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/infra/terraform/aws-bots/README.md).

## Environment Variables

See [`.env.example`](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/.env.example) for required keys.

Minimum:

- `OPENAI_API_KEY`
- `MANIFOLD_API_KEY_OPENAI`

Optional:

- `DATABASE_URL`

## Repository Map

- [agent/](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/agent) core agent, trading, DB, API server
- [web/predict-arena/](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/web/predict-arena) dashboard frontend
- [infra/terraform/aws-bots/](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/infra/terraform/aws-bots) AWS infrastructure
- [scripts/](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/scripts) operational scripts
- [db/init/](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/db/init) SQL schema

## Showcase Notes

- This project targets Manifold (play-money Mana), not real-money brokerage execution.
- HTTPS + custom domain is supported via ALB + ACM + Route53.
- Default prod template still supports direct URL mode for quickest launch.
- Pre-share checklist: [docs/showcase-checklist.md](/Users/alastairgrant/Personal/Code_Projects/Agent-Time/docs/showcase-checklist.md).
