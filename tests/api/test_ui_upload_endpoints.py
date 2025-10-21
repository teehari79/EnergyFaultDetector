"""Tests covering the lightweight UI-focused API endpoints."""

from __future__ import annotations

import io
from typing import Generator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")

from fastapi.testclient import TestClient  # noqa: E402  pylint: disable=wrong-import-position

import energy_fault_detector.api.app as ui_app  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def _reset_prediction_store() -> Generator[None, None, None]:
    """Ensure each test starts with a clean prediction cache."""

    ui_app.PREDICTION_STORE.clear()
    yield
    ui_app.PREDICTION_STORE.clear()


def _create_client() -> TestClient:
    return TestClient(ui_app.app)


def test_dataset_upload_creates_prediction_and_supporting_endpoints() -> None:
    client = _create_client()

    csv_payload = "time_stamp,value_a,value_b\n2024-01-01T00:00:00Z,1,5\n2024-01-01T01:00:00Z,3,10\n"
    files = {"file": ("sample.csv", io.BytesIO(csv_payload.encode("utf-8")), "text/csv")}
    data = {"batch_name": "Batch-42", "notes": "Integration test"}

    response = client.post("/api/predictions", files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    prediction_id = body["prediction_id"]
    assert prediction_id

    anomaly_response = client.get(
        "/webhooks/anomalies",
        params={"prediction_id": prediction_id},
        headers={"accept": "application/json"},
    )
    assert anomaly_response.status_code == 200
    anomalies = anomaly_response.json()
    assert isinstance(anomalies, list)
    assert all("metric" in event for event in anomalies)

    narrative_response = client.post("/api/narratives", json={"prediction_id": prediction_id})
    assert narrative_response.status_code == 200
    assert "summary" in narrative_response.json()

    report_response = client.get(f"/api/predictions/{prediction_id}/report")
    assert report_response.status_code == 200
    assert report_response.headers["content-type"].startswith("text/csv")
    assert "metric" in report_response.text
