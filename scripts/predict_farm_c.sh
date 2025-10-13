#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:-http://127.0.0.1:8000/predict}"
PAYLOAD_FILE="${2:-docs/examples/farm_c_prediction.json}"

if [ ! -f "${PAYLOAD_FILE}" ]; then
  echo "Payload file '${PAYLOAD_FILE}' was not found." >&2
  exit 1
fi

curl \
  --silent \
  --show-error \
  --fail \
  -X POST "${API_URL}" \
  -H 'Content-Type: application/json' \
  --data "@${PAYLOAD_FILE}" | jq
