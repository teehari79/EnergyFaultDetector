#!/usr/bin/env bash
set -euo pipefail

# Change to the repository root (the directory containing this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Allow callers to override the configuration file location. When not provided
# the default service configuration bundled with the package is used.
CONFIG_PATH="${1:-energy_fault_detector/api/service_config.yaml}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Configuration file '${CONFIG_PATH}' was not found." >&2
  exit 1
fi

export EFD_SERVICE_CONFIG="${CONFIG_PATH}"

# Uvicorn is the recommended ASGI server for FastAPI applications. This command
# exposes the API on http://127.0.0.1:8000 by default. Pass additional
# arguments (e.g. --host 0.0.0.0 --port 8080) after the configuration path to
# customise the listener address.
shift || true

exec uvicorn energy_fault_detector.api.app:app "$@"
