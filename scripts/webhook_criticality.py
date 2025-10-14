"""Run a CLI webhook receiver for critical event notifications."""

from webhook_server_base import run_cli


if __name__ == "__main__":
    run_cli("criticality", default_port=9012)
