# AWS Bot Infra (Terraform)

This stack provisions low-cost AWS infrastructure to run isolated bot hosts in EC2 (ChatGPT by default) plus a dedicated market-fetcher host.

## What This Creates

- 1x ECR repository for the bot Docker image
- 1x EC2 bot instance by default (`gpt`)
- 1x EC2 market-fetcher instance
- CloudWatch log groups for bots (one per configured bot)
- 1x CloudWatch log group for market fetcher
- 1x S3 bucket for shared market cache (`shared/shared_markets.json`)
- 1x IAM role + instance profile (SSM read, ECR pull, CloudWatch write, S3 cache read/write)
- 1x security group (no inbound by default)
- systemd service/timer on each host

## Prerequisites

- AWS credentials configured locally (`aws sts get-caller-identity` must work)
- Terraform `>= 1.5` installed as the HashiCorp CLI (`terraform version`)
- Docker + `buildx` installed locally
- `jq` (only needed for the optional one-cycle helper command)
- Default VPC/subnets in your chosen region

## Setup From Zero

### 1) Configure Terraform

```bash
cd infra/terraform/aws-bots
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` as needed.

To wire Supabase/Postgres into bot runtime, set:

```bash
database_url_param_name = "/agent-time/dev/DATABASE_URL"
require_database_url    = true
```

`require_database_url = true` makes bot startup fail fast if the DB secret is missing.

### 2) Create AWS Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

Useful outputs:

- `ecr_repository_url`
- `ssm_parameter_names`
- `bot_instance_ids`
- `bot_public_ips`
- `cloudwatch_log_groups`
- `market_fetcher_instance_id`
- `market_fetcher_log_group`
- `shared_market_cache_s3_uri`
- `database_url_ssm_parameter_name`

### 3) Build and Push Docker Image to ECR (amd64)

Run from repo root:

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=$(terraform -chdir=infra/terraform/aws-bots output -raw ecr_repository_url)

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker buildx build \
  --platform linux/amd64 \
  -t "${ECR_REPO}:latest" \
  --push \
  .
```

### 4) Write Secrets to SSM Parameter Store

Terraform outputs the required names in `ssm_parameter_names`.

Default parameter paths:

- `/agent-time/<env>/gpt/OPENAI_API_KEY`
- `/agent-time/<env>/gpt/MANIFOLD_API_KEY_OPENAI`

If you changed `project_name` or `environment`, use `terraform output -json ssm_parameter_names` and write exactly those paths.

Example:

```bash
aws ssm put-parameter \
  --region us-east-1 \
  --name "/agent-time/dev/gpt/OPENAI_API_KEY" \
  --type SecureString \
  --value "YOUR_OPENAI_KEY" \
  --overwrite
```

Repeat for both keys.

Optional shared DB secret (used by all bots):

```bash
aws ssm put-parameter \
  --region us-east-1 \
  --name "/agent-time/dev/DATABASE_URL" \
  --type SecureString \
  --value "postgresql://..." \
  --overwrite
```

If you newly enabled `database_url_param_name` in `terraform.tfvars`, run `terraform apply` again so instances pick up updated bootstrap config.

### 5) (Optional) Run One Cycle Manually

```bash
AWS_REGION=us-east-1
TF_DIR=infra/terraform/aws-bots

FETCHER_ID=$(terraform -chdir="${TF_DIR}" output -raw market_fetcher_instance_id)
BOT_IDS=$(terraform -chdir="${TF_DIR}" output -json bot_instance_ids | jq -r '.[]')

aws ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids "${FETCHER_ID}" \
  --document-name AWS-RunShellScript \
  --parameters commands='["sudo systemctl start agent-time-market-fetcher.service"]'

aws ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids ${BOT_IDS} \
  --document-name AWS-RunShellScript \
  --parameters commands='["sudo systemctl start agent-time-bot.service"]'
```

## Tear Everything Down

### 1) Destroy Terraform-managed resources

```bash
cd infra/terraform/aws-bots
terraform destroy -auto-approve
```

### 2) If destroy fails with `BucketNotEmpty`, empty bucket and retry

```bash
aws s3 rm s3://$(terraform output -raw shared_market_cache_bucket) --recursive
terraform destroy -auto-approve
```

Alternative: set `shared_market_cache_force_destroy = true` in `terraform.tfvars`.

### 3) Delete SSM secrets (they are created manually, not by Terraform)

```bash
AWS_REGION=us-east-1
PROJECT_NAME=agent-time
ENVIRONMENT=dev
for p in $(aws ssm get-parameters-by-path \
  --region "${AWS_REGION}" \
  --path "/${PROJECT_NAME}/${ENVIRONMENT}" \
  --recursive \
  --query 'Parameters[].Name' \
  --output text); do
  aws ssm delete-parameter --region "${AWS_REGION}" --name "$p"
done
```

### 4) Quick cleanup checks

`RepositoryNotFound` / empty responses are expected after full teardown.

```bash
AWS_REGION=us-east-1
PROJECT_NAME=agent-time
ENVIRONMENT=dev

aws ecr describe-repositories \
  --region "${AWS_REGION}" \
  --repository-names "${PROJECT_NAME}-${ENVIRONMENT}-bots"
aws logs describe-log-groups \
  --region "${AWS_REGION}" \
  --log-group-name-prefix "/${PROJECT_NAME}/${ENVIRONMENT}"
aws ssm get-parameters-by-path \
  --region "${AWS_REGION}" \
  --path "/${PROJECT_NAME}/${ENVIRONMENT}" \
  --recursive
```

## Notes

- `DATABASE_URL` is intentionally omitted for cloud-only bot testing.
- If `database_url_param_name` is not set, logs will show DB writes being skipped.
- Each instance runs exactly one bot agent.
- Wallet verification is enabled via `MANIFOLD_VERIFY_BEFORE_EACH_REQUEST=1` in `common_env`.
