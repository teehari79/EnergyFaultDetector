#!/usr/bin/env bash
# Utility script to launch a MongoDB instance for local development.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/setup_mongodb_local.sh [options]

Bootstraps a MongoDB instance in Docker for local development and testing.

Options (environment variables may also be used):
  -h, --help                 Show this help message and exit
  --container-name <name>    Name for the MongoDB container (default: energy-fault-detector-mongo)
  --port <port>              Host port to expose MongoDB on (default: 27017)
  --data-dir <path>          Host directory to persist MongoDB data (default: ../.mongodb/data)
  --log-dir <path>           Host directory to persist MongoDB logs (default: ../.mongodb/logs)
  --root-user <user>         MongoDB root username (default: efd_root)
  --root-password <pass>     MongoDB root password (default: efd_root_password)
  --app-db <db>              Application database name (default: energy_fault_detector)
  --app-user <user>          Application database user (default: efd_app)
  --app-password <pass>      Application user password (default: efd_app_password)

Environment overrides:
  MONGO_CONTAINER_NAME, MONGO_PORT, MONGO_DATA_DIR, MONGO_LOG_DIR,
  MONGO_INITDB_ROOT_USERNAME, MONGO_INITDB_ROOT_PASSWORD,
  MONGO_APP_DB, MONGO_APP_USER, MONGO_APP_PASSWORD
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONTAINER_NAME=${MONGO_CONTAINER_NAME:-energy-fault-detector-mongo}
PORT=${MONGO_PORT:-27017}
DATA_DIR=${MONGO_DATA_DIR:-"${ROOT_DIR}/.mongodb/data"}
LOG_DIR=${MONGO_LOG_DIR:-"${ROOT_DIR}/.mongodb/logs"}
ROOT_USER=${MONGO_INITDB_ROOT_USERNAME:-efd_root}
ROOT_PASSWORD=${MONGO_INITDB_ROOT_PASSWORD:-efd_root_password}
APP_DB=${MONGO_APP_DB:-energy_fault_detector}
APP_USER=${MONGO_APP_USER:-efd_app}
APP_PASSWORD=${MONGO_APP_PASSWORD:-efd_app_password}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --container-name)
      CONTAINER_NAME=$2
      shift 2
      ;;
    --port)
      PORT=$2
      shift 2
      ;;
    --data-dir)
      DATA_DIR=$2
      shift 2
      ;;
    --log-dir)
      LOG_DIR=$2
      shift 2
      ;;
    --root-user)
      ROOT_USER=$2
      shift 2
      ;;
    --root-password)
      ROOT_PASSWORD=$2
      shift 2
      ;;
    --app-db)
      APP_DB=$2
      shift 2
      ;;
    --app-user)
      APP_USER=$2
      shift 2
      ;;
    --app-password)
      APP_PASSWORD=$2
      shift 2
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

echo "Using container name: ${CONTAINER_NAME}"
echo "Exposing MongoDB on host port: ${PORT}"
echo "Persisting data in: ${DATA_DIR}"
echo "Persisting logs in: ${LOG_DIR}"
echo "Root user: ${ROOT_USER}"
echo "Application database: ${APP_DB}"
echo "Application user: ${APP_USER}"

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "Found existing container named ${CONTAINER_NAME}."
  if docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "Container is already running; skipping recreation."
  else
    echo "Removing stopped container ${CONTAINER_NAME}."
    docker rm "${CONTAINER_NAME}"
  fi
fi

if ! docker ps --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
  echo "Starting MongoDB container..."
  docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:27017" -v "${DATA_DIR}:/data/db" -v "${LOG_DIR}:/var/log/mongodb" -e MONGO_INITDB_ROOT_USERNAME="${ROOT_USER}" -e MONGO_INITDB_ROOT_PASSWORD="${ROOT_PASSWORD}" mongo:7.0 --wiredTigerCacheSizeGB 1 --logpath /var/log/mongodb/mongod.log --bind_ip_all >/dev/null
else
  echo "MongoDB container already running; skipping docker run."
fi

echo "Waiting for MongoDB to accept connections..."
ATTEMPTS=0
until docker exec "${CONTAINER_NAME}" mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
  ATTEMPTS=$((ATTEMPTS + 1))
  if [[ ${ATTEMPTS} -ge 20 ]]; then
    echo "MongoDB did not become ready in time." >&2
    exit 1
  fi
  sleep 1
done

echo "MongoDB is ready. Ensuring application user exists..."
CREATE_USER_SCRIPT="const targetDb = db.getSiblingDB('${APP_DB}');if (!targetDb.getUser('${APP_USER}')) { targetDb.createUser({ user: '${APP_USER}', pwd: '${APP_PASSWORD}', roles: [{ role: 'readWrite', db: '${APP_DB}' }] }); print('Created application user ${APP_USER} in database ${APP_DB}.'); } else { print('Application user ${APP_USER} already exists in database ${APP_DB}.'); }"

docker exec "${CONTAINER_NAME}" mongosh --quiet --username "${ROOT_USER}" --password "${ROOT_PASSWORD}" --authenticationDatabase admin --eval "${CREATE_USER_SCRIPT}" >/dev/null

cat <<INFO
MongoDB local setup complete!

Connection details:
  MongoDB URI (admin): mongodb://${ROOT_USER}:${ROOT_PASSWORD}@localhost:${PORT}/admin
  MongoDB URI (app):   mongodb://${APP_USER}:${APP_PASSWORD}@localhost:${PORT}/${APP_DB}

Remember to keep your credentials safe and rotate them for production usage.
INFO
