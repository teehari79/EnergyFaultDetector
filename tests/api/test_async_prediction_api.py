from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402  pylint: disable=wrong-import-position

from energy_fault_detector.api import prediction_api
from energy_fault_detector.api.prediction_api import (
    AsyncPredictionPayload,
    AuthenticationRequest,
    AuthenticationResponse,
    EventMetadata,
    EventSensorData,
    PredictionRequest,
    PredictionSuccessResponse,
    SensorPoint,
)
from energy_fault_detector.api.settings import get_settings


def _build_stub_response() -> PredictionSuccessResponse:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)

    event = EventMetadata(event_id=1, start=start, end=end, duration_seconds=300.0)
    sensor_point = SensorPoint(
        timestamp=start,
        anomaly_score=0.9,
        threshold_score=0.5,
        behaviour="anomaly",
        cumulative_anomaly_score=1,
        sensors={"sensor_a": 1.0},
    )
    event_sensor = EventSensorData(
        event_id=1,
        points=[sensor_point],
        arcana_mean_importances={"sensor_a": 1.0},
        arcana_losses=[],
    )

    return PredictionSuccessResponse(status="success", events=[event], event_sensor_data=[event_sensor])


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    prediction_api._AUTH_SESSIONS.clear()
    prediction_api._PREDICTION_JOBS.clear()
    yield
    prediction_api._AUTH_SESSIONS.clear()
    prediction_api._PREDICTION_JOBS.clear()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    stub_response = _build_stub_response()

    def _fake_run_prediction(_: PredictionRequest) -> PredictionSuccessResponse:
        return stub_response

    monkeypatch.setattr(prediction_api, "run_prediction", _fake_run_prediction)
    return TestClient(prediction_api.app)


def _encrypt_credentials(seed: str, username: str, password: str) -> str:
    credentials = {"username": username, "password": password}
    return prediction_api._encrypt_payload(seed, credentials, "auth_credentials")


def _encrypt_prediction_payload(seed: str, auth_hash: str, payload: Dict[str, object]) -> str:
    return prediction_api._encrypt_payload(seed, payload, auth_hash)


def test_authenticate_and_predict_flow(client: TestClient):
    settings = get_settings()
    tenant = settings.security.tenants["sample-org"]

    encrypted_credentials = _encrypt_credentials(tenant.seed_token, "analyst", "changeme123")

    auth_response = client.post(
        "/auth",
        json=AuthenticationRequest(
            organization_id="sample-org",
            credentials_encrypted=encrypted_credentials,
        ).dict(),
    )
    assert auth_response.status_code == 200
    auth_payload = AuthenticationResponse(**auth_response.json())

    auth_hash = prediction_api._hash_auth_token(auth_payload.auth_token, tenant.seed_token)
    prediction_payload = AsyncPredictionPayload(
        organization_id="sample-org",
        request=PredictionRequest(
            model_path="dummy",
            data=[{"time_stamp": datetime.now(timezone.utc).isoformat(), "sensor_a": 1.0}],
        ),
    ).dict()

    encrypted_payload = _encrypt_prediction_payload(tenant.seed_token, auth_hash, prediction_payload)

    predict_response = client.post(
        "/predict",
        json={
            "auth_token": auth_payload.auth_token,
            "auth_hash": auth_hash,
            "payload_encrypted": encrypted_payload,
        },
    )
    assert predict_response.status_code == 200
    job_id = predict_response.json()["job_id"]

    status_payload = None
    for _ in range(10):
        status_response = client.get(
            f"/jobs/{job_id}", params={"auth_token": auth_payload.auth_token}
        )
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert "anomaly_detection" in status_payload["steps"]
    assert status_payload["result"]["status"] == "success"


def test_prediction_rejects_invalid_hash(client: TestClient):
    settings = get_settings()
    tenant = settings.security.tenants["sample-org"]
    encrypted_credentials = _encrypt_credentials(tenant.seed_token, "analyst", "changeme123")

    auth_response = client.post(
        "/auth",
        json={"organization_id": "sample-org", "credentials_encrypted": encrypted_credentials},
    )
    assert auth_response.status_code == 200
    auth_token = auth_response.json()["auth_token"]

    payload = AsyncPredictionPayload(
        organization_id="sample-org",
        request=PredictionRequest(
            model_path="dummy",
            data=[{"time_stamp": datetime.now(timezone.utc).isoformat(), "sensor_a": 1.0}],
        ),
    ).dict()
    bogus_hash = "deadbeef"
    encrypted_payload = _encrypt_prediction_payload(tenant.seed_token, bogus_hash, payload)

    predict_response = client.post(
        "/predict",
        json={
            "auth_token": auth_token,
            "auth_hash": bogus_hash,
            "payload_encrypted": encrypted_payload,
        },
    )

    assert predict_response.status_code == 403
    detail = predict_response.json()
    assert detail["detail"]["code"] == "EFD_DECRYPTION_FAILED"


def test_job_status_requires_valid_token(client: TestClient):
    settings = get_settings()
    tenant = settings.security.tenants["sample-org"]
    encrypted_credentials = _encrypt_credentials(tenant.seed_token, "analyst", "changeme123")

    auth_response = client.post(
        "/auth",
        json={"organization_id": "sample-org", "credentials_encrypted": encrypted_credentials},
    )
    auth_token = auth_response.json()["auth_token"]

    auth_hash = prediction_api._hash_auth_token(auth_token, tenant.seed_token)
    payload = AsyncPredictionPayload(
        organization_id="sample-org",
        request=PredictionRequest(
            model_path="dummy",
            data=[{"time_stamp": datetime.now(timezone.utc).isoformat(), "sensor_a": 1.0}],
        ),
    ).dict()
    encrypted_payload = _encrypt_prediction_payload(tenant.seed_token, auth_hash, payload)

    predict_response = client.post(
        "/predict",
        json={
            "auth_token": auth_token,
            "auth_hash": auth_hash,
            "payload_encrypted": encrypted_payload,
        },
    )
    job_id = predict_response.json()["job_id"]

    # Query with wrong token
    status_response = client.get(f"/jobs/{job_id}", params={"auth_token": "invalid"})
    assert status_response.status_code == 401


def test_file_based_prediction_payload_is_normalised(monkeypatch, tmp_path):
    stub_response = _build_stub_response()
    captured: Dict[str, PredictionRequest] = {}

    def _fake_run_prediction(request: PredictionRequest) -> PredictionSuccessResponse:
        captured["request"] = request
        return stub_response

    monkeypatch.setattr(prediction_api, "run_prediction", _fake_run_prediction)

    model_directory = tmp_path / "models" / "farm-c" / "1.0.0"
    model_directory.mkdir(parents=True)

    class _StubRegistry:
        def resolve(self, model_name: str, model_version: Optional[str] = None):
            assert model_name == "farm-c"
            assert model_version == "1.0.0"
            return model_directory, "1.0.0"

    def _fake_registry() -> _StubRegistry:
        return _StubRegistry()

    monkeypatch.setattr(prediction_api, "_get_model_registry", _fake_registry)

    client = TestClient(prediction_api.app)
    settings = get_settings()
    tenant = settings.security.tenants["sample-org"]

    encrypted_credentials = _encrypt_credentials(tenant.seed_token, "analyst", "changeme123")
    auth_response = client.post(
        "/auth",
        json=AuthenticationRequest(
            organization_id="sample-org",
            credentials_encrypted=encrypted_credentials,
        ).dict(),
    )
    auth_payload = AuthenticationResponse(**auth_response.json())

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("time_stamp,sensor_a\n2024-01-01T00:00:00Z,1.0\n", encoding="utf-8")

    auth_hash = prediction_api._hash_auth_token(auth_payload.auth_token, tenant.seed_token)
    request_payload = {
        "organization_id": "sample-org",
        "request": {
            "model_name": "farm-c",
            "model_version": "1.0.0",
            "data_path": str(csv_path),
            "timestamp_column": "time_stamp",
            "enable_narrative": False,
        },
    }

    encrypted_payload = _encrypt_prediction_payload(tenant.seed_token, auth_hash, request_payload)

    predict_response = client.post(
        "/predict",
        json={
            "auth_token": auth_payload.auth_token,
            "auth_hash": auth_hash,
            "payload_encrypted": encrypted_payload,
        },
    )
    assert predict_response.status_code == 200

    job_id = predict_response.json()["job_id"]
    for _ in range(10):
        status_response = client.get(
            f"/jobs/{job_id}", params={"auth_token": auth_payload.auth_token}
        )
        assert status_response.status_code == 200
        if status_response.json()["status"] == "completed":
            break
        time.sleep(0.05)

    assert "request" in captured
    normalised = captured["request"]
    assert normalised.model_path == str(model_directory)
    assert normalised.timestamp_column == "time_stamp"
    assert len(normalised.data) == 1
    assert normalised.data[0]["sensor_a"] == 1.0

