# Showcase Checklist

Use this before sharing Agent-Time publicly.

## Technical Readiness

- `terraform apply` succeeds in `infra/terraform/aws-bots`.
- Bot timer is active: `agent-time-bot.timer`.
- Dashboard service is active: `agent-time-dashboard.service`.
- CloudWatch logs show a clean run with at least one decision cycle.
- Public dashboard URL resolves and loads data.
- HTTPS custom domain resolves with a valid ACM certificate.

## Security Basics

- Rotate API keys used during development.
- Confirm no secrets are committed (`.env`, API keys, DB URLs).
- Use least-privilege IAM credentials for deployment.

## Demo Artifacts

- Dashboard screenshot (portfolio summary + trade table).
- CloudWatch snippet showing scheduled run timestamp.
- Architecture image or one-slide diagram.
- Short written summary of:
  - problem
  - solution
  - architecture
  - measurable outcome

## Suggested LinkedIn Project Summary

Built **Agent-Time**, an autonomous Manifold trading system that runs daily on AWS, executes strategy decisions with LLM tooling, logs outcomes to persistent storage, and exposes a live web dashboard for performance review.
