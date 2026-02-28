#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${TF_DIR:-${ROOT_DIR}/infra/terraform/aws-bots}"
TFVARS_FILE="${TFVARS_FILE:-${TF_DIR}/terraform.tfvars}"
TFVARS_TEMPLATE="${TFVARS_TEMPLATE:-${TF_DIR}/terraform.tfvars.prod.example}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
AUTO_APPROVE="${AUTO_APPROVE:-false}"
SKIP_TERRAFORM_APPLY="${SKIP_TERRAFORM_APPLY:-false}"
SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD:-false}"
SKIP_SERVICE_RESTART="${SKIP_SERVICE_RESTART:-false}"
TF_OUTPUTS_JSON=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

start_services() {
  local fetcher_id
  fetcher_id="$(jq -r '.market_fetcher_instance_id.value // "null"' <<<"${TF_OUTPUTS_JSON}")"
  if [[ -n "${fetcher_id}" && "${fetcher_id}" != "null" ]]; then
    aws ssm send-command \
      --region "${AWS_REGION}" \
      --instance-ids "${fetcher_id}" \
      --document-name AWS-RunShellScript \
      --parameters commands='["sudo systemctl start agent-time-market-fetcher.service"]' \
      >/dev/null
  fi

  bot_ids=()
  while IFS= read -r bot_id; do
    [[ -z "${bot_id}" ]] && continue
    bot_ids+=("${bot_id}")
  done < <(jq -r '.bot_instance_ids.value | .[]' <<<"${TF_OUTPUTS_JSON}")
  if [[ "${#bot_ids[@]}" -eq 0 ]]; then
    echo "No bot instances found in Terraform output." >&2
    exit 1
  fi

  aws ssm send-command \
    --region "${AWS_REGION}" \
    --instance-ids "${bot_ids[@]}" \
    --document-name AWS-RunShellScript \
    --parameters commands='["sudo systemctl start agent-time-bot.service","sudo systemctl restart agent-time-dashboard.service"]' \
    >/dev/null
}

check_required_ssm_params() {
  local missing=()
  while IFS= read -r param_name; do
    [[ -z "${param_name}" ]] && continue
    if ! aws ssm get-parameter --region "${AWS_REGION}" --name "${param_name}" >/dev/null 2>&1; then
      missing+=("${param_name}")
    fi
  done < <(jq -r '.ssm_parameter_names.value | to_entries[] | .value.llm_api_key, .value.manifold_api_key' <<<"${TF_OUTPUTS_JSON}")

  db_param="$(jq -r '.database_url_ssm_parameter_name.value // empty' <<<"${TF_OUTPUTS_JSON}")"
  if [[ -n "${db_param}" && "${db_param}" != "null" ]]; then
    if ! aws ssm get-parameter --region "${AWS_REGION}" --name "${db_param}" >/dev/null 2>&1; then
      missing+=("${db_param}")
    fi
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Missing required SSM parameters:" >&2
    printf ' - %s\n' "${missing[@]}" >&2
    echo "Create these first, then rerun this script." >&2
    exit 1
  fi
}

load_tf_outputs() {
  if ! TF_OUTPUTS_JSON="$(terraform -chdir="${TF_DIR}" output -json)"; then
    echo "Failed to load Terraform outputs. Run terraform apply first." >&2
    exit 1
  fi

  if ! jq -e 'type == "object" and length > 0' <<<"${TF_OUTPUTS_JSON}" >/dev/null; then
    echo "Terraform state has no outputs. Run terraform apply first." >&2
    exit 1
  fi

  for key in ecr_repository_url ssm_parameter_names bot_instance_ids; do
    if ! jq -e --arg key "${key}" 'has($key) and .[$key].value != null' <<<"${TF_OUTPUTS_JSON}" >/dev/null; then
      echo "Missing required Terraform output '${key}'. Run terraform apply to populate state." >&2
      exit 1
    fi
  done
}

require_cmd aws
require_cmd terraform
require_cmd docker
require_cmd jq

if [[ ! -f "${TFVARS_FILE}" ]]; then
  cp "${TFVARS_TEMPLATE}" "${TFVARS_FILE}"
  echo "Created ${TFVARS_FILE} from ${TFVARS_TEMPLATE}"
fi

if grep -Eq 'dashboard_domain_name\s*=\s*"agent\.yourdomain\.com"' "${TFVARS_FILE}" || \
   grep -Eq 'dashboard_hosted_zone_name\s*=\s*"yourdomain\.com"' "${TFVARS_FILE}"; then
  echo "Update dashboard_domain_name/dashboard_hosted_zone_name in ${TFVARS_FILE} before deploying HTTPS." >&2
  exit 1
fi

terraform -chdir="${TF_DIR}" init

if ! is_true "${SKIP_TERRAFORM_APPLY}"; then
  if is_true "${AUTO_APPROVE}"; then
    terraform -chdir="${TF_DIR}" apply -auto-approve
  else
    terraform -chdir="${TF_DIR}" apply
  fi
fi

load_tf_outputs
check_required_ssm_params

if ! is_true "${SKIP_IMAGE_BUILD}"; then
  account_id="$(aws sts get-caller-identity --query Account --output text)"
  ecr_repo="$(jq -r '.ecr_repository_url.value' <<<"${TF_OUTPUTS_JSON}")"

  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com"

  docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    -t "${ecr_repo}:${IMAGE_TAG}" \
    --push \
    "${ROOT_DIR}"
fi

if ! is_true "${SKIP_SERVICE_RESTART}"; then
  start_services
fi

echo
echo "Deployment complete."
echo "Public dashboard URL(s):"
"${ROOT_DIR}/scripts/dashboard_url.sh"
