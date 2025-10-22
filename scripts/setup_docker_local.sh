#!/usr/bin/env bash
# Utility script to launch the Energy Fault Detector stack in Docker.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/setup_docker_local.sh [options] [-- <docker-compose up args>]

Bootstraps the Energy Fault Detector API, Node service, React UI, and MongoDB
using Docker Compose. Options allow overriding container names, ports, and
volume locations. Extra arguments after "--" are forwarded to
"docker compose up".

Options (environment variables may also be used):
  -h, --help                      Show this help message and exit
  --project-name <name>           Docker Compose project name (default: energy-fault-detector)
  --compose-file <path>           Path to the docker compose file (default: ./docker/docker-compose.local.yml)
  --mongo-container-name <name>   MongoDB container name (default: energy-fault-detector-mongo)
  --mongo-port <port>             Host port for MongoDB (default: 27017)
  --mongo-data-dir <path>         Directory for MongoDB data volume (default: ./.mongodb/data)
  --mongo-log-dir <path>          Directory for MongoDB log volume (default: ./.mongodb/logs)
  --mongo-root-user <user>        MongoDB root username (default: efd_root)
  --mongo-root-password <pass>    MongoDB root password (default: efd_root_password)
  --mongo-app-db <name>           MongoDB application database (default: energy_fault_detector)
  --mongo-app-user <user>         MongoDB application username (default: efd_app)
  --mongo-app-password <pass>     MongoDB application password (default: efd_app_password)
  --api-container-name <name>     API container name (default: energy-fault-detector-api)
  --api-port <port>               Host port for the synchronous FastAPI service (default: 8000)
  --prediction-port <port>        Host port for the asynchronous prediction API (default: 8001)
  --results-dir <path>            Directory to persist API results (default: ./results)
  --models-dir <path>             Directory containing trained models (default: ./models)
  --web-service-container <name>  Node service container name (default: energy-fault-detector-web-service)
  --web-service-port <port>       Host port for the Node service (default: 4000)
  --web-ui-container <name>       React UI container name (default: energy-fault-detector-web-ui)
  --web-ui-port <port>            Host port for the React UI (default: 5173)

Environment overrides:
  EFD_COMPOSE_PROJECT, EFD_COMPOSE_FILE, MONGO_CONTAINER_NAME, MONGO_PORT,
  MONGO_DATA_DIR, MONGO_LOG_DIR, MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD,
  MONGO_APP_DB, MONGO_APP_USER, MONGO_APP_PASSWORD, API_CONTAINER_NAME,
  API_PORT, PREDICTION_PORT, HOST_RESULTS_DIR, HOST_MODELS_DIR,
  WEB_SERVICE_CONTAINER_NAME, WEB_SERVICE_PORT,
  WEB_UI_CONTAINER_NAME, WEB_UI_PORT

Examples:
  ./scripts/setup_docker_local.sh
  ./scripts/setup_docker_local.sh --api-port 9000 -- --build --detach
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

PROJECT_NAME=${EFD_COMPOSE_PROJECT:-energy-fault-detector}
COMPOSE_FILE=${EFD_COMPOSE_FILE:-"${ROOT_DIR}/docker/docker-compose.local.yml"}
MONGO_CONTAINER_NAME=${MONGO_CONTAINER_NAME:-energy-fault-detector-mongo}
MONGO_PORT=${MONGO_PORT:-27017}
MONGO_DATA_DIR=${MONGO_DATA_DIR:-"${ROOT_DIR}/.mongodb/data"}
MONGO_LOG_DIR=${MONGO_LOG_DIR:-"${ROOT_DIR}/.mongodb/logs"}
MONGO_ROOT_USER=${MONGO_INITDB_ROOT_USERNAME:-${MONGO_ROOT_USER:-efd_root}}
MONGO_ROOT_PASSWORD=${MONGO_INITDB_ROOT_PASSWORD:-${MONGO_ROOT_PASSWORD:-efd_root_password}}
MONGO_APP_DB=${MONGO_APP_DB:-energy_fault_detector}
MONGO_APP_USER=${MONGO_APP_USER:-efd_app}
MONGO_APP_PASSWORD=${MONGO_APP_PASSWORD:-efd_app_password}
API_CONTAINER_NAME=${API_CONTAINER_NAME:-energy-fault-detector-api}
API_PORT=${API_PORT:-8000}
PREDICTION_PORT=${PREDICTION_PORT:-8001}
HOST_RESULTS_DIR=${HOST_RESULTS_DIR:-"${ROOT_DIR}/results"}
HOST_MODELS_DIR=${HOST_MODELS_DIR:-"${ROOT_DIR}/models"}
WEB_SERVICE_CONTAINER_NAME=${WEB_SERVICE_CONTAINER_NAME:-energy-fault-detector-web-service}
WEB_SERVICE_PORT=${WEB_SERVICE_PORT:-4000}
WEB_UI_CONTAINER_NAME=${WEB_UI_CONTAINER_NAME:-energy-fault-detector-web-ui}
WEB_UI_PORT=${WEB_UI_PORT:-5173}

COMPOSE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --project-name)
      PROJECT_NAME=$2
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE=$2
      shift 2
      ;;
    --mongo-container-name)
      MONGO_CONTAINER_NAME=$2
      shift 2
      ;;
    --mongo-port)
      MONGO_PORT=$2
      shift 2
      ;;
    --mongo-data-dir)
      MONGO_DATA_DIR=$2
      shift 2
      ;;
    --mongo-log-dir)
      MONGO_LOG_DIR=$2
      shift 2
      ;;
    --mongo-root-user)
      MONGO_ROOT_USER=$2
      shift 2
      ;;
    --mongo-root-password)
      MONGO_ROOT_PASSWORD=$2
      shift 2
      ;;
    --mongo-app-db)
      MONGO_APP_DB=$2
      shift 2
      ;;
    --mongo-app-user)
      MONGO_APP_USER=$2
      shift 2
      ;;
    --mongo-app-password)
      MONGO_APP_PASSWORD=$2
      shift 2
      ;;
    --api-container-name)
      API_CONTAINER_NAME=$2
      shift 2
      ;;
    --api-port)
      API_PORT=$2
      shift 2
      ;;
    --prediction-port)
      PREDICTION_PORT=$2
      shift 2
      ;;
    --results-dir)
      HOST_RESULTS_DIR=$2
      shift 2
      ;;
    --models-dir)
      HOST_MODELS_DIR=$2
      shift 2
      ;;
    --web-service-container)
      WEB_SERVICE_CONTAINER_NAME=$2
      shift 2
      ;;
    --web-service-port)
      WEB_SERVICE_PORT=$2
      shift 2
      ;;
    --web-ui-container)
      WEB_UI_CONTAINER_NAME=$2
      shift 2
      ;;
    --web-ui-port)
      WEB_UI_PORT=$2
      shift 2
      ;;
    --)
      shift
      COMPOSE_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required but was not found in PATH." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD=(docker-compose)
else
  echo "Error: docker compose plugin or docker-compose binary is required." >&2
  exit 1
fi

abs_path() {
  python - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

COMPOSE_FILE=$(abs_path "${COMPOSE_FILE}")
MONGO_DATA_DIR=$(abs_path "${MONGO_DATA_DIR}")
MONGO_LOG_DIR=$(abs_path "${MONGO_LOG_DIR}")
HOST_RESULTS_DIR=$(abs_path "${HOST_RESULTS_DIR}")
HOST_MODELS_DIR=$(abs_path "${HOST_MODELS_DIR}")

mkdir -p "${MONGO_DATA_DIR}" "${MONGO_LOG_DIR}" "${HOST_RESULTS_DIR}" "${HOST_MODELS_DIR}"

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Error: docker compose file '${COMPOSE_FILE}' not found." >&2
  exit 1
fi

echo "Using docker compose file: ${COMPOSE_FILE}"
echo "Project name: ${PROJECT_NAME}"
echo "MongoDB container: ${MONGO_CONTAINER_NAME} (port ${MONGO_PORT})"
echo "API container: ${API_CONTAINER_NAME} (ports ${API_PORT}/${PREDICTION_PORT})"
echo "Node service container: ${WEB_SERVICE_CONTAINER_NAME} (port ${WEB_SERVICE_PORT})"
echo "Web UI container: ${WEB_UI_CONTAINER_NAME} (port ${WEB_UI_PORT})"

default_args=(--build -d)
if [[ ${#COMPOSE_ARGS[@]} -eq 0 ]]; then
  COMPOSE_ARGS=("${default_args[@]}")
fi

env \
  COMPOSE_FILE="${COMPOSE_FILE}" \
  COMPOSE_PROJECT_NAME="${PROJECT_NAME}" \
  MONGO_CONTAINER_NAME="${MONGO_CONTAINER_NAME}" \
  MONGO_PORT="${MONGO_PORT}" \
  MONGO_DATA_DIR="${MONGO_DATA_DIR}" \
  MONGO_LOG_DIR="${MONGO_LOG_DIR}" \
  MONGO_ROOT_USER="${MONGO_ROOT_USER}" \
  MONGO_INITDB_ROOT_USERNAME="${MONGO_ROOT_USER}" \
  MONGO_ROOT_PASSWORD="${MONGO_ROOT_PASSWORD}" \
  MONGO_INITDB_ROOT_PASSWORD="${MONGO_ROOT_PASSWORD}" \
  MONGO_APP_DB="${MONGO_APP_DB}" \
  MONGO_APP_USER="${MONGO_APP_USER}" \
  MONGO_APP_PASSWORD="${MONGO_APP_PASSWORD}" \
  API_CONTAINER_NAME="${API_CONTAINER_NAME}" \
  API_PORT="${API_PORT}" \
  PREDICTION_PORT="${PREDICTION_PORT}" \
  HOST_RESULTS_DIR="${HOST_RESULTS_DIR}" \
  HOST_MODELS_DIR="${HOST_MODELS_DIR}" \
  WEB_SERVICE_CONTAINER_NAME="${WEB_SERVICE_CONTAINER_NAME}" \
  WEB_SERVICE_PORT="${WEB_SERVICE_PORT}" \
  WEB_UI_CONTAINER_NAME="${WEB_UI_CONTAINER_NAME}" \
  WEB_UI_PORT="${WEB_UI_PORT}" \
  "${DOCKER_COMPOSE_CMD[@]}" up "${COMPOSE_ARGS[@]}"

echo
env \
  COMPOSE_FILE="${COMPOSE_FILE}" \
  COMPOSE_PROJECT_NAME="${PROJECT_NAME}" \
  "${DOCKER_COMPOSE_CMD[@]}" ps

echo
cat <<INFO
Stack is up! Services are reachable at:
  FastAPI (sync):        http://localhost:${API_PORT}
  FastAPI (prediction):  http://localhost:${PREDICTION_PORT}
  Node proxy service:    http://localhost:${WEB_SERVICE_PORT}
  React UI:              http://localhost:${WEB_UI_PORT}
  MongoDB:               mongodb://localhost:${MONGO_PORT}

Use "${DOCKER_COMPOSE_CMD[*]} -p ${PROJECT_NAME} down" to stop the stack.
INFO
