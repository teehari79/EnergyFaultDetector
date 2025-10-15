#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

WEBHOOK_SCRIPTS=(
  "webhook_anomalies.py"
  "webhook_events.py"
  "webhook_criticality.py"
  "webhook_root_cause.py"
  "webhook_narrative.py"
)

PIDS=()
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "\nStopping webhook listeners..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT

for script in "${WEBHOOK_SCRIPTS[@]}"; do
  echo "Starting ${script}"
  python "${script}" "$@" &
  PIDS+=($!)
  sleep 0.2
done

echo "Webhook listeners are running. Press Ctrl+C to stop."
wait -n
