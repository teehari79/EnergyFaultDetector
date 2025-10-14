"""Utilities for running simple CLI webhook receivers for the prediction API."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class _ServerConfig:
    host: str
    port: int
    step_name: str


def _create_handler(step_name: str) -> type:
    """Create a request handler that logs webhook payloads to stdout."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "EFDWebhook/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - match BaseHTTPRequestHandler API
            logging.info("%s - - %s", self.client_address[0], format % args)

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                decoded_body: Any = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError:
                decoded_body = raw_body.decode("utf-8", errors="replace")

            auth_header = self.headers.get("X-EFD-Auth-Token", "<missing>")
            job_header = self.headers.get("X-EFD-Job-ID", "<missing>")

            logging.info("Received %s webhook", step_name)
            logging.info("Auth token: %s", auth_header)
            logging.info("Job ID: %s", job_header)
            if isinstance(decoded_body, dict):
                pretty = json.dumps(decoded_body, indent=2)
                suffix = ""
            else:
                pretty = decoded_body
                suffix = " (raw)"

            logging.info("Payload:%s\n%s", suffix, pretty)

            response = json.dumps({"status": "acknowledged", "step": step_name}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return _Handler


def _serve(config: _ServerConfig) -> None:
    handler = _create_handler(config.step_name)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    logging.info(
        "Listening for %s webhook events on http://%s:%s/",
        config.step_name,
        config.host,
        config.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual CLI usage
        logging.info("Shutting down %s webhook listener", config.step_name)
    finally:
        server.server_close()


def run_cli(step_name: str, default_port: int) -> None:
    """Entry-point helper for webhook scripts."""

    parser = argparse.ArgumentParser(
        description=(
            "Start a lightweight HTTP server that prints received '%s' webhook payloads."
            % step_name
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port to listen on (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    _serve(_ServerConfig(host=args.host, port=args.port, step_name=step_name))


__all__ = ["run_cli"]
