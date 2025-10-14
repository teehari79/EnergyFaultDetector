"""Run a CLI webhook receiver for narrative generation summaries."""

from webhook_server_base import run_cli


if __name__ == "__main__":
    run_cli("narrative", default_port=9014)
