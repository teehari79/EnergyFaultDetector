from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import pytest

pytest.importorskip("pandas")
pytest.importorskip("fastapi")

import pandas as pd  # noqa: E402  pylint: disable=wrong-import-position
from fastapi.testclient import TestClient  # noqa: E402  pylint: disable=wrong-import-position

from energy_fault_detector.api import prediction_api
from energy_fault_detector.api.model_registry import ModelRegistry
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


def _create_model_dir(base: Path, model_name: str, version: str) -> Path:
    path = base / model_name / version
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model: test", encoding="utf-8")
    return path


def _create_asset_dir(base: Path, asset_number: str, version: str) -> Path:
    path = base / f"asset_{asset_number}" / "models" / version
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model: test", encoding="utf-8")
    return path


def test_resolve_model_path_prefers_model_name(tmp_path, monkeypatch):
    registry = ModelRegistry(root_directory=tmp_path)
    model_dir = _create_model_dir(tmp_path, "farm-c", "1.0.0")

    monkeypatch.setattr(prediction_api, "_get_model_registry", lambda: registry)

    resolved = prediction_api._resolve_model_path("farm-c", "1.0.0", asset_name="farm-c-asset-34")

    assert resolved == str(model_dir)


def test_resolve_model_path_falls_back_to_asset_directory(tmp_path, monkeypatch):
    registry = ModelRegistry(root_directory=tmp_path)
    asset_dir = _create_asset_dir(tmp_path, "34", "20240101_000000")

    monkeypatch.setattr(prediction_api, "_get_model_registry", lambda: registry)

    resolved = prediction_api._resolve_model_path("missing", None, asset_name="farm-c-asset-34")

    assert resolved == str(asset_dir)


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


def test_load_file_prediction_data_detects_delimiter(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "time_stamp;sensor_a;sensor_b\n"
        "2024-01-01T00:00:00Z;1.0;0.5\n"
        "2024-01-01T00:01:00Z;2.0;0.6\n",
        encoding="utf-8",
    )

    records = prediction_api._load_file_prediction_data(str(csv_path))

    assert len(records) == 2
    assert records[0] == {
        "time_stamp": "2024-01-01T00:00:00Z",
        "sensor_a": 1.0,
        "sensor_b": 0.5,
    }


def test_load_file_prediction_data_falls_back_when_no_delimiter(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("time_stamp\n2024-01-01T00:00:00Z\n", encoding="utf-8")

    records = prediction_api._load_file_prediction_data(str(csv_path))

    assert records == [{"time_stamp": "2024-01-01T00:00:00Z"}]


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

    def events_factory(
        sensor_data: pd.DataFrame,
        boolean_information: pd.Series,
        min_event_length: Optional[int],
        min_event_duration: Optional[Union[str, float, int]],
    ):  # pylint: disable=unused-argument
        assert min_event_length == 1
        assert min_event_duration is None
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


def test_run_prediction_applies_sensor_mapping(tmp_path):
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"]
    )

    sensor_values = pd.DataFrame(
        {
            "sensor_1_avg": [1.0, 2.0, 3.0],
            "sensor_1_min": [0.5, 0.4, 0.3],
            "sensor_2_avg": [5.0, 5.5, 5.2],
        },
        index=timestamps,
    )

    predicted_anomalies = pd.DataFrame(
        {
            "anomaly": [False, True, True],
            "behaviour": ["normal", "anomaly", "anomaly"],
            "anamoly_score": [0.1, 0.9, 1.2],
            "threshold_score": [0.8, 0.8, 0.8],
            "cumulative_anamoly_score": [0, 1, 2],
        },
        index=timestamps,
    )

    result = FaultDetectionResult(
        predicted_anomalies=predicted_anomalies,
        reconstruction=sensor_values,
        recon_error=sensor_values,
        anomaly_score=pd.DataFrame({"value": [0.1, 0.9, 1.2]}, index=timestamps),
        bias_data=None,
        arcana_losses=None,
        tracked_bias=None,
    )

    detector = _create_detector(result, tuple(sensor_values.columns))

    request = PredictionRequest(
        model_path=str(tmp_path),
        data=[
            {
                "time_stamp": ts.isoformat(),
                "sensor_1_avg": float(idx + 1),
                "sensor_1_min": float(idx),
                "sensor_2_avg": 5.0 + 0.5 * idx,
            }
            for idx, ts in enumerate(timestamps)
        ],
        timestamp_column="time_stamp",
        min_event_length=1,
    )

    sensor_mapping = pd.DataFrame(
        {
            "sensor name": ["sensor_1", "sensor_2"],
            "feature description": ["Nacelle temperature", "Hydraulic pressure"],
        }
    )
    sensor_mapping.to_csv(tmp_path / "sensor_data.csv", index=False)

    event_data = sensor_values.iloc[1:]
    event_meta = pd.DataFrame(
        {
            "start": [event_data.index[0]],
            "end": [event_data.index[-1]],
            "duration": [event_data.index[-1] - event_data.index[0]],
        }
    )

    def events_factory(*_, **__):
        return event_meta, [event_data]

    def event_analyzer(*_, **__):
        importances = pd.Series({"sensor_1_avg": 0.7, "sensor_2_avg": 0.3})
        losses = pd.DataFrame(
            {"sensor_1_avg": [0.2, 0.1], "sensor_2_avg": [0.15, 0.05]},
            index=[0, 1],
        )
        losses.index.name = "Iteration"
        return importances, losses

    response = run_prediction(
        request,
        detector_loader=lambda _: detector,
        events_factory=events_factory,
        event_analyzer=event_analyzer,
    )

    point = response.event_sensor_data[0].points[0]
    assert set(point.sensors) == {
        "Nacelle temperature (avg)",
        "Nacelle temperature (min)",
        "Hydraulic pressure (avg)",
    }
    assert response.event_sensor_data[0].arcana_mean_importances == {
        "Nacelle temperature (avg)": 0.7,
        "Hydraulic pressure (avg)": 0.3,
    }
    assert response.event_sensor_data[0].arcana_losses is not None
    assert all(
        "Nacelle temperature (avg)" in record and "Hydraulic pressure (avg)" in record
        for record in response.event_sensor_data[0].arcana_losses
    )


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
