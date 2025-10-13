from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energy_fault_detector.api import prediction_api
from energy_fault_detector.api.prediction_api import (
    DataTypeMismatchError,
    EmptyInputError,
    InvalidModelError,
    PredictionRequest,
    PredictionSuccessResponse,
    SchemaMismatchError,
    TimestampValidationError,
    run_prediction,
)
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult


def _create_detector(result: FaultDetectionResult, expected_columns: Tuple[str, ...]):
    column_selector = SimpleNamespace(feature_names_out_=list(expected_columns))
    imputer = SimpleNamespace(feature_names_in_=list(expected_columns))
    data_preprocessor = SimpleNamespace(
        named_steps={"column_selector": column_selector, "imputer": imputer},
        get_feature_names_out=lambda: list(expected_columns),
    )

    class _Detector:
        data_preprocessor = data_preprocessor

        @staticmethod
        def predict(sensor_data: pd.DataFrame, root_cause_analysis: bool = False):  # pylint: disable=unused-argument
            return result

    return _Detector()


def _sample_fault_detection_result(timestamps: pd.DatetimeIndex) -> FaultDetectionResult:
    sensor_values = pd.DataFrame(
        {
            "sensor_a": [1.0, 2.0, 3.0],
            "sensor_b": [0.5, 0.6, 0.7],
        },
        index=timestamps,
    )

    predicted_anomalies = pd.DataFrame(
        {
            "anomaly": [False, True, True],
            "behaviour": ["normal", "anamoly", "anamoly"],
            "anamoly_score": [0.1, 0.9, 1.1],
            "threshold_score": [0.8, 0.8, 0.8],
            "cumulative_anamoly_score": [0, 1, 2],
        },
        index=timestamps,
    )

    anomaly_score = pd.DataFrame({"value": [0.1, 0.9, 1.1]}, index=timestamps)

    return FaultDetectionResult(
        predicted_anomalies=predicted_anomalies,
        reconstruction=sensor_values,
        recon_error=sensor_values,
        anomaly_score=anomaly_score,
        bias_data=None,
        arcana_losses=None,
        tracked_bias=None,
    )


def test_run_prediction_success():
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"]
    )
    result = _sample_fault_detection_result(timestamps)

    detector = _create_detector(result, ("sensor_a", "sensor_b"))

    request = PredictionRequest(
        model_path="dummy",
        data=[
            {"time_stamp": ts.isoformat(), "sensor_a": float(idx + 1), "sensor_b": 0.5 + idx * 0.1}
            for idx, ts in enumerate(timestamps)
        ],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    event_data = result.reconstruction.iloc[1:]
    event_meta = pd.DataFrame(
        {
            "start": [event_data.index[0]],
            "end": [event_data.index[-1]],
            "duration": [event_data.index[-1] - event_data.index[0]],
        }
    )

    def events_factory(sensor_data: pd.DataFrame, boolean_information: pd.Series, min_event_length: int):  # pylint: disable=unused-argument
        return event_meta, [event_data]

    def event_analyzer(detector_obj, data, track_losses):  # pylint: disable=unused-argument
        importances = pd.Series({"sensor_a": 0.6, "sensor_b": 0.4})
        losses = pd.DataFrame(
            {"Combined Loss": [0.2, 0.1], "Reconstruction Loss": [0.15, 0.05], "Regularization Loss": [0.05, 0.05]},
            index=[0, 50],
        )
        losses.index.name = "Iteration"
        return importances, losses

    response = run_prediction(
        request,
        detector_loader=lambda _: detector,
        events_factory=events_factory,
        event_analyzer=event_analyzer,
    )

    assert isinstance(response, PredictionSuccessResponse)
    assert response.status == "success"
    assert len(response.events) == 1
    assert response.events[0].event_id == 1
    assert len(response.event_sensor_data) == 1
    assert len(response.event_sensor_data[0].points) == len(event_data)
    assert response.event_sensor_data[0].arcana_mean_importances == {"sensor_a": 0.6, "sensor_b": 0.4}


def test_run_prediction_schema_mismatch():
    timestamps = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])
    result = _sample_fault_detection_result(timestamps)
    detector = _create_detector(result, ("sensor_a", "sensor_b"))

    request = PredictionRequest(
        model_path="dummy",
        data=[
            {"time_stamp": ts.isoformat(), "sensor_a": float(idx + 1)}
            for idx, ts in enumerate(timestamps)
        ],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    with pytest.raises(SchemaMismatchError):
        run_prediction(request, detector_loader=lambda _: detector, events_factory=lambda *args, **kwargs: (pd.DataFrame(), []))


def test_run_prediction_dtype_mismatch():
    timestamps = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])
    result = _sample_fault_detection_result(timestamps)
    detector = _create_detector(result, ("sensor_a", "sensor_b"))

    request = PredictionRequest(
        model_path="dummy",
        data=[
            {"time_stamp": ts.isoformat(), "sensor_a": float(idx + 1), "sensor_b": "bad"}
            for idx, ts in enumerate(timestamps)
        ],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    with pytest.raises(DataTypeMismatchError):
        run_prediction(request, detector_loader=lambda _: detector, events_factory=lambda *args, **kwargs: (pd.DataFrame(), []))


def test_run_prediction_timestamp_validation():
    request = PredictionRequest(
        model_path="dummy",
        data=[{"sensor_a": 1.0, "sensor_b": 2.0}],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    with pytest.raises(TimestampValidationError):
        run_prediction(request, detector_loader=lambda _: None, events_factory=lambda *args, **kwargs: (pd.DataFrame(), []))


def test_run_prediction_invalid_model():
    timestamps = pd.to_datetime(["2024-01-01T00:00:00Z"])
    request = PredictionRequest(
        model_path="dummy",
        data=[{"time_stamp": timestamps[0].isoformat(), "sensor_a": 1.0, "sensor_b": 2.0}],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    def failing_loader(_: str):
        raise InvalidModelError("model missing")

    with pytest.raises(InvalidModelError):
        run_prediction(request, detector_loader=failing_loader, events_factory=lambda *args, **kwargs: (pd.DataFrame(), []))


def test_run_prediction_empty_payload():
    request = PredictionRequest.construct(
        model_path="dummy",
        data=[],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    with pytest.raises(EmptyInputError):
        run_prediction(request, detector_loader=lambda _: None, events_factory=lambda *args, **kwargs: (pd.DataFrame(), []))


def test_predict_endpoint_success(monkeypatch):
    response_payload = PredictionSuccessResponse(status="success", events=[], event_sensor_data=[])
    monkeypatch.setattr(prediction_api, "run_prediction", lambda req: response_payload)
    client = TestClient(prediction_api.app)

    payload = {
        "model_path": "dummy",
        "data": [{"time_stamp": "2024-01-01T00:00:00Z", "sensor_a": 1.0, "sensor_b": 2.0}],
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_predict_endpoint_error(monkeypatch):
    def failing_run_prediction(req):  # pylint: disable=unused-argument
        raise SchemaMismatchError("missing columns")

    monkeypatch.setattr(prediction_api, "run_prediction", failing_run_prediction)
    client = TestClient(prediction_api.app)

    payload = {
        "model_path": "dummy",
        "data": [{"time_stamp": "2024-01-01T00:00:00Z", "sensor_a": 1.0}],
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "EFD_SCHEMA_MISMATCH"
    assert detail["status"] == "error"
