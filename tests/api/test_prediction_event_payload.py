from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import pandas as pd

from energy_fault_detector.api.prediction_api import _build_event_sensor_payload


def test_build_event_sensor_payload_handles_duplicate_prediction_index():
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"], utc=True
    )
    event_data = pd.DataFrame(
        {
            "sensor_a": [1.0, 2.0],
            "sensor_b": [0.5, 0.7],
        },
        index=timestamps,
    )

    prediction_index = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:01:00Z",
        ],
        utc=True,
    )
    predicted_anomalies = pd.DataFrame(
        {
            "anamoly_score": [0.2, 0.9, 0.3],
            "threshold_score": [0.4, 0.4, 0.6],
            "behaviour": ["nominal", "anomaly", "nominal"],
            "cumulative_anamoly_score": [1, 2, 3],
        },
        index=prediction_index,
    )

    payload = _build_event_sensor_payload(
        event_id=1,
        event_data=event_data,
        predicted_anomalies=predicted_anomalies,
        arcana_mean_importances=None,
        arcana_losses=None,
        sensor_name_mapping={},
    )

    assert len(payload.points) == len(event_data)
    first_point = payload.points[0]
    assert first_point.anomaly_score == 0.9
    assert first_point.behaviour == "anomaly"
