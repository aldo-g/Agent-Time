# AWS Bot Infra (Terraform)

Terraform stack for running Agent-Time in AWS with:
- one ChatGPT bot EC2 instance (`gpt`)
- optional dedicated market-fetcher EC2 instance
- shared S3 market cache
- ECR image hosting
- CloudWatch logs
- optional publishable dashboard/API service on the bot host
- optional HTTPS + domain via ALB + ACM + Route53

## Fastest Launch (Script)

From repo root:

```bash
./scripts/deploy_aws_prod.sh
```

HTTPS + domain profile:

```bash
TFVARS_TEMPLATE=infra/terraform/aws-bots/terraform.tfvars.prod.https.example ./scripts/deploy_aws_prod.sh
```

Print dashboard URL again later:

```bash
./scripts/dashboard_url.sh
```

## What To Do (Quick Path)

1. Copy `terraform.tfvars.example`, `terraform.tfvars.prod.example`, or `terraform.tfvars.prod.https.example` to `terraform.tfvars`.
2. Run `terraform init && terraform apply`.
3. Build and push the Docker image to ECR.
4. Create SSM secrets for `OPENAI_API_KEY` and `MANIFOLD_API_KEY_OPENAI`.
5. Start bot service once via SSM (and fetcher if enabled).
6. Watch CloudWatch logs to confirm successful runs.

## Prerequisites

- AWS credentials configured locally (`aws sts get-caller-identity` works)
- Terraform >= 1.5
- Docker with Buildx
- `jq`
- Default VPC/subnets available in the target region

## 1) Configure Terraform

```bash
cd infra/terraform/aws-bots
cp terraform.tfvars.prod.https.example terraform.tfvars
```

Then edit `terraform.tfvars` and replace:
- `dashboard_domain_name`
- either `dashboard_route53_zone_id` or `dashboard_hosted_zone_name`

Recommended minimal settings in `terraform.tfvars`:

```hcl
aws_region                 = "eu-west-1"
environment                = "prod"
create_public_eips         = true
default_schedule           = "*-*-* 12:00:00 UTC"
create_market_fetcher_instance = false
enable_shared_market_cache = false
bot_skip_market_fetch      = false
enable_dashboard_service   = true
dashboard_bind_address     = "0.0.0.0"
dashboard_ingress_cidr     = null
enable_dashboard_alb       = true
dashboard_domain_name      = "agent.yourdomain.com"
dashboard_hosted_zone_name = "yourdomain.com"
```

Optional DB wiring (if you want run/trade writes to Postgres):

```hcl
database_url_param_name = "/agent-time/prod/DATABASE_URL"
require_database_url    = true
```

Notes for HTTPS/domain:
- Use either `dashboard_route53_zone_id` or `dashboard_hosted_zone_name`.
- If you already have a certificate, set `dashboard_certificate_arn` and Terraform will skip ACM certificate creation.

## 2) Create Infrastructure

```bash
cd infra/terraform/aws-bots
terraform init
terraform plan
terraform apply
```

Useful outputs:
- `ecr_repository_url`
- `ssm_parameter_names`
- `bot_instance_ids`
- `market_fetcher_instance_id`
- `cloudwatch_log_groups`
- `market_fetcher_log_group`
- `shared_market_cache_s3_uri`
- `dashboard_public_urls`
- `dashboard_alb_dns_name`
- `dashboard_https_url`
- `dashboard_domain_name`

## 3) Build And Push Image

Run from repo root:

```bash
AWS_REGION=eu-west-1
TF_DIR=infra/terraform/aws-bots
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=$(terraform -chdir="${TF_DIR}" output -raw ecr_repository_url)

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker buildx build \
  --platform linux/amd64 \
  -t "${ECR_REPO}:latest" \
  --push \
  .
```

## 4) Create Required SSM Secrets

Get the exact parameter names Terraform expects:

```bash
AWS_REGION=eu-west-1
TF_DIR=infra/terraform/aws-bots
terraform -chdir="${TF_DIR}" output -json ssm_parameter_names | jq
```

Default names:
- `/agent-time/<env>/gpt/OPENAI_API_KEY`
- `/agent-time/<env>/gpt/MANIFOLD_API_KEY_OPENAI`

Set them:

```bash
aws ssm put-parameter \
  --region "${AWS_REGION}" \
  --name "/agent-time/prod/gpt/OPENAI_API_KEY" \
  --type SecureString \
  --value "YOUR_OPENAI_KEY" \
  --overwrite

aws ssm put-parameter \
  --region "${AWS_REGION}" \
  --name "/agent-time/prod/gpt/MANIFOLD_API_KEY_OPENAI" \
  --type SecureString \
  --value "YOUR_MANIFOLD_KEY" \
  --overwrite
```

Optional DB secret:

```bash
aws ssm put-parameter \
  --region "${AWS_REGION}" \
  --name "/agent-time/prod/DATABASE_URL" \
  --type SecureString \
  --value "postgresql://..." \
  --overwrite
```

If you changed DB-related tfvars after initial apply, run `terraform apply` again.

## 5) Run Once Manually (Recommended After First Setup)

```bash
AWS_REGION=eu-west-1
TF_DIR=infra/terraform/aws-bots

FETCHER_ID=$(terraform -chdir="${TF_DIR}" output -raw market_fetcher_instance_id)
BOT_IDS=$(terraform -chdir="${TF_DIR}" output -json bot_instance_ids | jq -r '.[]')

if [ "${FETCHER_ID}" != "null" ]; then
  aws ssm send-command \
    --region "${AWS_REGION}" \
    --instance-ids "${FETCHER_ID}" \
    --document-name AWS-RunShellScript \
    --parameters commands='["sudo systemctl start agent-time-market-fetcher.service"]'
fi

aws ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids ${BOT_IDS} \
  --document-name AWS-RunShellScript \
  --parameters commands='["sudo systemctl start agent-time-bot.service","sudo systemctl restart agent-time-dashboard.service"]'
```

## 6) Verify It Is Working

Get log group names from Terraform outputs:

```bash
AWS_REGION=eu-west-1
TF_DIR=infra/terraform/aws-bots
terraform -chdir="${TF_DIR}" output -json cloudwatch_log_groups | jq
terraform -chdir="${TF_DIR}" output -raw market_fetcher_log_group
```

Tail bot logs:

```bash
BOT_LOG_GROUP=$(terraform -chdir="${TF_DIR}" output -json cloudwatch_log_groups | jq -r '.gpt')
aws logs tail "${BOT_LOG_GROUP}" --region "${AWS_REGION}" --follow
```

Tail market fetcher logs:

```bash
FETCHER_LOG_GROUP=$(terraform -chdir="${TF_DIR}" output -raw market_fetcher_log_group)
if [ "${FETCHER_LOG_GROUP}" != "null" ]; then
  aws logs tail "${FETCHER_LOG_GROUP}" --region "${AWS_REGION}" --follow
fi
```

## Public Dashboard Access (Publishable)

Get the preferred URL:

```bash
TF_DIR=infra/terraform/aws-bots
terraform -chdir="${TF_DIR}" output -raw dashboard_https_url
```

If `dashboard_https_url` is null, fallback to direct instance URL:

```bash
terraform -chdir="${TF_DIR}" output -json dashboard_public_urls | jq
```

ALB DNS (useful while waiting for DNS propagation):

```bash
terraform -chdir="${TF_DIR}" output -raw dashboard_alb_dns_name
```

## Updating The Deployment

When app code changes:

1. Build/push image again with the same tag (`latest`) or update `image_tag`.
2. Restart services via SSM to pull/run the new image:

```bash
AWS_REGION=eu-west-1
TF_DIR=infra/terraform/aws-bots
FETCHER_ID=$(terraform -chdir="${TF_DIR}" output -raw market_fetcher_instance_id)
BOT_IDS=$(terraform -chdir="${TF_DIR}" output -json bot_instance_ids | jq -r '.[]')

if [ "${FETCHER_ID}" != "null" ]; then
  aws ssm send-command \
    --region "${AWS_REGION}" \
    --instance-ids "${FETCHER_ID}" \
    --document-name AWS-RunShellScript \
    --parameters commands='["sudo systemctl start agent-time-market-fetcher.service"]'
fi

aws ssm send-command \
  --region "${AWS_REGION}" \
  --instance-ids ${BOT_IDS} \
  --document-name AWS-RunShellScript \
  --parameters commands='["sudo systemctl start agent-time-bot.service","sudo systemctl restart agent-time-dashboard.service"]'
```

## Tear Down

### 1) Destroy Terraform resources

```bash
cd infra/terraform/aws-bots
terraform destroy -auto-approve
```

### 2) If destroy fails on non-empty S3 bucket

```bash
aws s3 rm s3://$(terraform -chdir=infra/terraform/aws-bots output -raw shared_market_cache_bucket) --recursive
terraform destroy -auto-approve
```

### 3) Delete SSM parameters (manual secrets)

```bash
AWS_REGION=eu-west-1
PROJECT_NAME=agent-time
ENVIRONMENT=prod
for p in $(aws ssm get-parameters-by-path \
  --region "${AWS_REGION}" \
  --path "/${PROJECT_NAME}/${ENVIRONMENT}" \
  --recursive \
  --query 'Parameters[].Name' \
  --output text); do
  aws ssm delete-parameter --region "${AWS_REGION}" --name "$p"
done
```

## Notes

- Timers are enabled by default (`enable_timers = true`), so services continue running on schedule.
- If `database_url_param_name` is unset, DB writes are skipped by design and dashboard falls back to JSONL files.
- For lowest monthly cost, use `create_market_fetcher_instance = false`, `enable_shared_market_cache = false`, and `bot_skip_market_fetch = false`.
