#!/usr/bin/env bash
set -euo pipefail

AGENT_NAME="${AGENT_NAME:-${1:-}}"
if [[ -z "${AGENT_NAME}" ]]; then
  echo "AGENT_NAME is required (env or first arg)."
  exit 1
fi

CACHE_PATH="${PREDICT_ARENA_MARKET_CACHE:-/data/shared_markets.json}"
WAIT_SECONDS="${MARKET_CACHE_WAIT_SECONDS:-120}"
SLEEP_INTERVAL="${MARKET_CACHE_WAIT_INTERVAL_SECONDS:-2}"

elapsed=0
while [[ ! -f "${CACHE_PATH}" ]]; do
  if (( elapsed >= WAIT_SECONDS )); then
    echo "Cache file not found after ${WAIT_SECONDS}s: ${CACHE_PATH}"
    break
  fi
  sleep "${SLEEP_INTERVAL}"
  elapsed=$((elapsed + SLEEP_INTERVAL))
done

VERBOSE_FLAG=()
if [[ "${AGENT_VERBOSE:-}" =~ ^(1|true|yes)$ ]]; then
  VERBOSE_FLAG=(--verbose)
fi

python -m agent.multi_runner \
  --agent "${AGENT_NAME}" \
  --skip-market-fetch \
  --market-cache "${CACHE_PATH}" \
  --max-attempts "${AGENT_MAX_ATTEMPTS:-2}" \
  --results "${AGENT_RESULTS_PATH:-/results/gpt_runs.jsonl}" \
  "${VERBOSE_FLAG[@]}"
