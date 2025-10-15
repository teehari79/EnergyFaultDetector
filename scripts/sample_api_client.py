"""Example CLI client that authenticates, encrypts payloads and triggers predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import httpx

# Ensure the local project package is importable even when the script is executed
# from within the ``scripts`` directory. Without this adjustment Python may resolve
# ``energy_fault_detector`` to an installed site-package instead of the local
# sources, leading to mismatched versions during development.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_fault_detector.api import prediction_api


def _load_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - user supplied payload
        raise SystemExit(f"Failed to parse payload JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("Prediction payload JSON must be an object.")

    data_path = payload.get("data_path")
    if isinstance(data_path, str) and data_path:
        csv_path = Path(data_path)
        if not csv_path.is_absolute():
            resolved = (path.parent / csv_path).resolve()
            payload["data_path"] = str(resolved)

    return payload


def _parse_webhook_overrides(values: Iterable[str]) -> Dict[str, str]:
    hooks: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"Invalid webhook override '{item}'. Use the form name=url.")
        name, url = item.split("=", 1)
        name = name.strip().lower()
        if not name:
            raise SystemExit(f"Webhook override '{item}' is missing a name.")
        hooks[name] = url.strip()
    return hooks


def _encrypt_credentials(seed_token: str, username: str, password: str) -> str:
    credentials = {"username": username, "password": password}
    return prediction_api._encrypt_payload(seed_token, credentials, "auth_credentials")


def _encrypt_prediction_payload(
    seed_token: str,
    auth_hash: str,
    payload: Mapping[str, Any],
) -> str:
    return prediction_api._encrypt_payload(seed_token, dict(payload), auth_hash)


def authenticate(
    client: httpx.Client,
    base_url: str,
    organization_id: str,
    seed_token: str,
    username: str,
    password: str,
) -> str:
    """Authenticate against the API and return an auth token."""

    encrypted_credentials = _encrypt_credentials(seed_token, username, password)
    try:
        response = client.post(
            f"{base_url}/auth",
            json={
                "organization_id": organization_id,
                "credentials_encrypted": encrypted_credentials,
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - depends on API availability
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 404:
            message = (
                "Authentication endpoint not found at "
                f"'{base_url}/auth'. Ensure the asynchronous prediction API is "
                "running (uvicorn energy_fault_detector.api.prediction_api:app) "
                "or update --base-url to point at the correct service."
            )
        else:
            detail: str
            try:
                detail = json.dumps(exc.response.json())
            except Exception:  # pragma: no cover - defensive, server dependent
                detail = exc.response.text if exc.response is not None else ""
            message = (
                f"Authentication failed with HTTP status {status}."
                f" Response details: {detail.strip()}"
            )
        raise SystemExit(message) from exc
    payload = response.json()
    return str(payload["auth_token"])


def submit_prediction(
    client: httpx.Client,
    base_url: str,
    organization_id: str,
    seed_token: str,
    auth_token: str,
    payload: Mapping[str, Any],
    webhooks: Optional[Mapping[str, str]] = None,
) -> str:
    """Encrypt and submit a prediction payload."""

    request_payload: Dict[str, Any] = {
        "organization_id": organization_id,
        "request": dict(payload),
    }
    if webhooks:
        request_payload["webhooks"] = webhooks

    auth_hash = prediction_api._hash_auth_token(auth_token, seed_token)
    encrypted_payload = _encrypt_prediction_payload(seed_token, auth_hash, request_payload)

    try:
        response = client.post(
            f"{base_url}/predict",
            json={
                "auth_token": auth_token,
                "auth_hash": auth_hash,
                "payload_encrypted": encrypted_payload,
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - depends on API availability
        status = exc.response.status_code if exc.response is not None else "unknown"
        try:
            detail = json.dumps(exc.response.json())
        except Exception:  # pragma: no cover - defensive, server dependent
            detail = exc.response.text if exc.response is not None else ""
        message = (
            f"Prediction request rejected with HTTP status {status}."
            f" Response details: {detail.strip()}"
        )
        raise SystemExit(message) from exc
    body = response.json()
    return str(body["job_id"])


def fetch_job_status(client: httpx.Client, base_url: str, job_id: str, auth_token: str) -> Dict[str, Any]:
    try:
        response = client.get(
            f"{base_url}/jobs/{job_id}",
            params={"auth_token": auth_token},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - depends on API availability
        status = exc.response.status_code if exc.response is not None else "unknown"
        try:
            detail = json.dumps(exc.response.json())
        except Exception:  # pragma: no cover - defensive, server dependent
            detail = exc.response.text if exc.response is not None else ""
        message = (
            f"Failed to retrieve job status with HTTP status {status}."
            f" Response details: {detail.strip()}"
        )
        raise SystemExit(message) from exc
    return response.json()


def _poll_until_complete(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    auth_token: str,
    interval: float,
) -> Dict[str, Any]:
    while True:
        status = fetch_job_status(client, base_url, job_id, auth_token)
        state = status.get("status", "unknown")
        print(f"Job {job_id} status: {state}")
        if state in {"completed", "failed"}:
            return status
        time.sleep(interval)


def _normalise_base_url(url: str) -> str:
    return url.rstrip("/")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Root URL of the API service")
    parser.add_argument("--organization", default="sample-org", help="Tenant identifier")
    parser.add_argument("--seed-token", default="sample-seed-token", help="Shared seed token for encryption")
    parser.add_argument("--username", default="analyst", help="API username")
    parser.add_argument("--password", default="changeme123", help="API password")
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path("docs/examples/farm_c_prediction.json"),
        help="Path to a JSON payload describing the prediction request",
    )
    parser.add_argument(
        "--webhook",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Optional webhook overrides (can be provided multiple times)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll the job status until completion",
    )
    parser.add_argument(
        "--enable-narrative",
        action="store_true",
        help=(
            "Include the enable_narrative flag in the request payload."
            " Narrative generation is disabled by default for the sample client."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between status checks when --poll is used",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_url = _normalise_base_url(args.base_url)
    payload = _load_payload(args.payload)

    # The asynchronous pipeline defaults to generating LLM narratives unless the request
    # explicitly opts out. For the sample client we disable the narrative step unless the
    # caller opts in via --enable-narrative to keep the example lightweight.
    payload = dict(payload)
    payload["enable_narrative"] = bool(args.enable_narrative)
    if not args.enable_narrative:
        print(
            "Narrative generation disabled for this request. "
            "Pass --enable-narrative to opt back in."
        )
    webhooks = _parse_webhook_overrides(args.webhook)

    with httpx.Client() as client:
        try:
            print(f"Authenticating against {base_url}/auth ...")
            auth_token = authenticate(
                client,
                base_url,
                organization_id=args.organization,
                seed_token=args.seed_token,
                username=args.username,
                password=args.password,
            )
            print("Authentication succeeded. Auth token issued.")

            if webhooks:
                print("Using webhook endpoints:")
                for name, url in webhooks.items():
                    print(f"  - {name}: {url}")

            print("Submitting encrypted prediction payload ...")
            job_id = submit_prediction(
                client,
                base_url,
                organization_id=args.organization,
                seed_token=args.seed_token,
                auth_token=auth_token,
                payload=payload,
                webhooks=webhooks or None,
            )
            print(f"Prediction job accepted with ID: {job_id}")

            if args.poll:
                print("Polling job status ...")
                status = _poll_until_complete(
                    client,
                    base_url,
                    job_id,
                    auth_token,
                    interval=args.poll_interval,
                )
                print("Final job status:")
                print(json.dumps(status, indent=2))
            else:
                print("Use the /jobs endpoint to query progress:"
                      f" curl '{base_url}/jobs/{job_id}?auth_token={auth_token}'")
        except httpx.RequestError as exc:  # pragma: no cover - depends on network configuration
            raise SystemExit(
                f"Failed to communicate with '{exc.request.url}'. {exc}"
            ) from exc

    return 0


if __name__ == "__main__":
    sys.exit(main())
