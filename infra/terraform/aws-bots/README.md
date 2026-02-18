# AWS Bot Infra (Terraform)

This stack provisions low-cost AWS infrastructure to run the three bots in separate EC2 instances (separate egress identity/IP per bot), without DB or frontend.

## What It Creates

- 1x ECR repository for your bot image
- 3x CloudWatch log groups (one per bot)
- 1x CloudWatch log group for dedicated market fetcher
- 1x S3 bucket for shared market cache snapshots
- 1x security group (no inbound by default)
- 1x IAM role + instance profile for EC2 (SSM, ECR pull, CloudWatch logs write, S3 shared-cache read/write)
- 3x EC2 instances (`gpt`, `claude`, `gemini`) in the default VPC
- 1x EC2 instance (`market-fetcher`) to fetch markets and upload shared cache
- Optional Elastic IPs per instance
- systemd service + timer on each instance to run one bot
- Shared-market flow: dedicated fetcher refreshes markets and uploads to S3; all bots download the same snapshot before each run

## Prerequisites

- AWS account + credentials configured locally
- Terraform `>= 1.5`
- Docker locally for building/pushing the image
- Default VPC exists in the selected region

## 1) Configure Terraform

```bash
cd infra/terraform/aws-bots
cp terraform.tfvars.example terraform.tfvars
```

Adjust values in `terraform.tfvars` as needed.

## 2) Deploy Infra

```bash
terraform init
terraform plan
terraform apply
```

Save outputs:

- `ecr_repository_url`
- `ssm_parameter_names`
- `bot_public_ips`
- `cloudwatch_log_groups`
- `market_fetcher_instance_id`
- `market_fetcher_public_ip`
- `market_fetcher_log_group`
- `shared_market_cache_bucket`
- `shared_market_cache_s3_uri`

## 3) Push App Image to ECR

Run from repo root:

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/agent-time-dev-bots"

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t agent-time:latest .
docker tag agent-time:latest "${ECR_REPO}:latest"
docker push "${ECR_REPO}:latest"
```

If you changed `project_name` / `environment`, use `terraform output -raw ecr_repository_url` instead of hardcoding.

## 4) Write Required Secrets to SSM Parameter Store

Parameter names are output by Terraform (`ssm_parameter_names`).  
By default they follow:

- `/agent-time/<env>/gpt/OPENAI_API_KEY`
- `/agent-time/<env>/gpt/MANIFOLD_API_KEY_OPENAI`
- `/agent-time/<env>/claude/CLAUDE_API_KEY`
- `/agent-time/<env>/claude/MANIFOLD_API_KEY_CLAUDE`
- `/agent-time/<env>/gemini/GEMINI_API_KEY`
- `/agent-time/<env>/gemini/MANIFOLD_API_KEY_GEMINI`

Example:

```bash
aws ssm put-parameter \
  --name "/agent-time/dev/gpt/OPENAI_API_KEY" \
  --type SecureString \
  --value "YOUR_OPENAI_KEY" \
  --overwrite
```

Repeat for all six parameters.

## 5) Verify Bot Hosts

Use SSM Session Manager or SSH.

Check timer and logs:

```bash
sudo systemctl status agent-time-bot.timer
sudo systemctl status agent-time-bot.service
sudo journalctl -u agent-time-bot.service -n 200 --no-pager
```

Container logs are in separate CloudWatch groups:

- `/${project_name}/${environment}/bots/gpt`
- `/${project_name}/${environment}/bots/claude`
- `/${project_name}/${environment}/bots/gemini`

Shared market cache object:

- `s3://<shared_market_cache_bucket>/<shared_market_cache_object_key>`

By default the dedicated `market-fetcher` instance runs on the schedule in `market_fetcher_schedule`, uploads a fresh snapshot to S3, and the bot instances wait for/download that object before running.

## Notes

- `DATABASE_URL` is intentionally omitted for cloud bot-only testing.
- Each instance runs one agent (`gpt-runner`, `claude-runner`, `gemini-runner`).
- For strict wallet checks, this assumes your current app code with `MANIFOLD_VERIFY_BEFORE_EACH_REQUEST=1`.
