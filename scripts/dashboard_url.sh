#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${TF_DIR:-${ROOT_DIR}/infra/terraform/aws-bots}"
PLAIN_OUTPUT="${1:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd terraform
require_cmd jq

if ! OUTPUTS_JSON="$(terraform -chdir="${TF_DIR}" output -json 2>/dev/null)"; then
  echo "Terraform outputs unavailable. Run terraform apply first." >&2
  exit 1
fi

if ! jq -e 'type == "object" and length > 0' <<<"${OUTPUTS_JSON}" >/dev/null; then
  echo "Terraform state has no outputs. Run terraform apply first." >&2
  exit 1
fi

HTTPS_URL_JSON="$(terraform -chdir="${TF_DIR}" output -json dashboard_https_url 2>/dev/null || echo "null")"
HTTPS_URL="$(jq -r 'select(type == "string")' <<<"${HTTPS_URL_JSON}" || true)"
if [[ -n "${HTTPS_URL}" && "${HTTPS_URL}" != "null" ]]; then
  if [[ "${PLAIN_OUTPUT}" == "--plain" ]]; then
    echo "${HTTPS_URL}"
  else
    echo "dashboard: ${HTTPS_URL}"
  fi
  exit 0
fi

URLS_JSON="$(terraform -chdir="${TF_DIR}" output -json dashboard_public_urls)"
if [[ "${URLS_JSON}" == "{}" ]]; then
  echo "No public dashboard URL is configured. Enable dashboard in terraform.tfvars first." >&2
  exit 1
fi

if [[ "${PLAIN_OUTPUT}" == "--plain" ]]; then
  jq -r '.gpt // (to_entries[0].value // empty)' <<<"${URLS_JSON}"
  exit 0
fi

jq -r 'to_entries[] | "\(.key): \(.value)"' <<<"${URLS_JSON}"
