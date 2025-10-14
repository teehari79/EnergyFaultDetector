"""Run a CLI webhook receiver for anomaly summaries."""

from webhook_server_base import run_cli


if __name__ == "__main__":
    run_cli("anomalies", default_port=9010)
