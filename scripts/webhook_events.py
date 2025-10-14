"""Run a CLI webhook receiver for event metadata updates."""

from webhook_server_base import run_cli


if __name__ == "__main__":
    run_cli("events", default_port=9011)
