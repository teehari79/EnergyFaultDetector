"""Run a CLI webhook receiver for root cause analysis results."""

from webhook_server_base import run_cli


if __name__ == "__main__":
    run_cli("root_cause", default_port=9013)
