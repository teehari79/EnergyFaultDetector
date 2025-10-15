#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${1:-energy_fault_detector/api/service_config.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration file '${CONFIG_PATH}' was not found." >&2
  exit 1
fi

CONFIG_PATH="$(python - "${CONFIG_PATH}" <<'PY'
import os
import sys

path = os.path.abspath(sys.argv[1])
print(path)
PY
)"

export EFD_SERVICE_CONFIG="${CONFIG_PATH}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
PREDICTION_HOST="${PREDICTION_API_HOST:-0.0.0.0}"
PREDICTION_PORT="${PREDICTION_API_PORT:-8001}"

UVICORN_ARGS=()
if (( $# > 1 )); then
  UVICORN_ARGS=("${@:2}")
fi

PIDS=()
start_uvicorn() {
  local app="$1"
  local host="$2"
  local port="$3"
  shift 3 || true

  if (( $# > 0 )); then
    uvicorn "$app" --host "$host" --port "$port" "$@" &
  else
    uvicorn "$app" --host "$host" --port "$port" &
  fi
  PIDS+=($!)
}

cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "\nStopping services..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT

echo "Starting synchronous API on ${API_HOST}:${API_PORT}"
start_uvicorn "energy_fault_detector.api.app:app" "${API_HOST}" "${API_PORT}" "${UVICORN_ARGS[@]}"

echo "Starting asynchronous prediction API on ${PREDICTION_HOST}:${PREDICTION_PORT}"
start_uvicorn "energy_fault_detector.api.prediction_api:app" "${PREDICTION_HOST}" "${PREDICTION_PORT}" "${UVICORN_ARGS[@]}"

echo "Services are running. Press Ctrl+C to stop."
wait -n
