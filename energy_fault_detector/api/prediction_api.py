"""FastAPI application exposing the quick fault detector prediction endpoint."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from energy_fault_detector._logs import setup_logging
from energy_fault_detector.config import Config
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.quick_fault_detection.data_loading import get_sensor_data
from energy_fault_detector.quick_fault_detection.quick_fault_detector import analyze_event
from energy_fault_detector.utils.analysis import create_events

DEFAULT_TIMESTAMP_COLUMN = "time_stamp"
DEFAULT_MIN_EVENT_LENGTH = 18

LOGGING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "logging.yaml"
setup_logging(str(LOGGING_CONFIG_PATH))

logger = logging.getLogger("energy_fault_detector.prediction_api")


class PredictionAPIError(Exception):
    """Base class for handled prediction API exceptions."""

    status_code: int = 500
    error_code: str = "EFD_INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidModelError(PredictionAPIError):
    """Raised when a referenced model cannot be loaded."""

    status_code = 404
    error_code = "EFD_MODEL_NOT_FOUND"


class SchemaMismatchError(PredictionAPIError):
    """Raised when the input schema does not match the model expectations."""

    status_code = 409
    error_code = "EFD_SCHEMA_MISMATCH"


class DataTypeMismatchError(PredictionAPIError):
    """Raised when data types of provided columns do not match the trained model."""

    status_code = 422
    error_code = "EFD_DATATYPE_MISMATCH"


class TimestampValidationError(PredictionAPIError):
    """Raised when the timestamp column is missing or cannot be parsed."""

    status_code = 422
    error_code = "EFD_TIMESTAMP_INVALID"


class EmptyInputError(PredictionAPIError):
    """Raised when the provided payload does not contain any rows."""

    status_code = 400
    error_code = "EFD_EMPTY_PAYLOAD"


class EventMetadata(BaseModel):
    """Metadata describing an anomaly event."""

    event_id: int = Field(..., description="Identifier of the anomaly event.")
    start: datetime = Field(..., description="Start timestamp of the event.")
    end: datetime = Field(..., description="End timestamp of the event.")
    duration_seconds: float = Field(..., ge=0, description="Duration of the event in seconds.")


class SensorPoint(BaseModel):
    """Time-series point describing sensor values and anomaly scores."""

    timestamp: datetime = Field(..., description="Timestamp of the sensor snapshot.")
    anomaly_score: Optional[float] = Field(None, description="Computed anomaly score.")
    threshold_score: Optional[float] = Field(None, description="Threshold used for anomaly decision.")
    behaviour: Optional[str] = Field(None, description="Model assessment of the behaviour at this timestamp.")
    cumulative_anomaly_score: Optional[int] = Field(None, description="Cumulative anomaly counter.")
    sensors: Dict[str, Optional[float]] = Field(..., description="Sensor readings contributing to the event.")


class EventSensorData(BaseModel):
    """Detailed sensor time series for an event, including ARCANA summaries."""

    event_id: int
    points: List[SensorPoint]
    arcana_mean_importances: Dict[str, float] = Field(default_factory=dict)
    arcana_losses: Optional[List[Dict[str, Any]]] = None


class PredictionSuccessResponse(BaseModel):
    """Successful prediction payload."""

    status: str = Field("success", const=True)
    events: List[EventMetadata]
    event_sensor_data: List[EventSensorData]


class PredictionRequest(BaseModel):
    """Payload describing a prediction request."""

    model_path: str = Field(..., description="Path to the saved fault detector model.")
    data: List[Dict[str, Any]] = Field(..., min_items=1, description="Prediction data as list of records.")
    timestamp_column: str = Field(
        DEFAULT_TIMESTAMP_COLUMN,
        description="Name of the timestamp column in the provided data.",
    )
    min_event_length: int = Field(
        DEFAULT_MIN_EVENT_LENGTH,
        ge=1,
        description="Minimum number of consecutive anomalies required to form an event.",
    )


def _load_fault_detector(model_path: str) -> FaultDetector:
    """Instantiate and load a :class:`FaultDetector` from disk."""

    try:
        fallback_config_path = Path(__file__).resolve().parent.parent / "base_config.yaml"
        fallback_config = Config(str(fallback_config_path))
        detector = FaultDetector(config=fallback_config)
        detector.load_models(model_path=model_path)
    except FileNotFoundError as exc:  # pragma: no cover - dependent on filesystem
        raise InvalidModelError(
            f"The provided model path '{model_path}' does not exist."
        ) from exc
    except (OSError, ValueError) as exc:  # pragma: no cover - dependent on filesystem
        raise InvalidModelError(
            f"Failed to load model artefacts from '{model_path}'."
        ) from exc
    return detector


def _ensure_datetime(value: Any) -> datetime:
    """Convert various timestamp representations to :class:`datetime`."""

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return pd.to_datetime(value).to_pydatetime()


def _to_optional_float(value: Any) -> Optional[float]:
    """Convert numeric values to floats while preserving missing values."""

    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_expected_columns(data_preprocessor: Any) -> List[str]:
    """Retrieve the expected input columns from the saved data preprocessor."""

    if data_preprocessor is None:
        return []

    named_steps = getattr(data_preprocessor, "named_steps", {})
    column_selector = named_steps.get("column_selector")
    if column_selector is not None and hasattr(column_selector, "feature_names_out_"):
        return list(column_selector.feature_names_out_)

    imputer = named_steps.get("imputer")
    if imputer is not None and hasattr(imputer, "feature_names_in_"):
        return list(imputer.feature_names_in_)

    if hasattr(data_preprocessor, "get_feature_names_out"):
        try:
            return list(data_preprocessor.get_feature_names_out())
        except ValueError:  # pragma: no cover - depends on sklearn internals
            pass

    if hasattr(data_preprocessor, "feature_names_in_"):
        return list(data_preprocessor.feature_names_in_)

    return []


def _validate_input_schema(
    sensor_data: pd.DataFrame,
    expected_columns: Sequence[str],
    original_columns: Iterable[str],
) -> None:
    """Validate that provided sensor data matches model expectations."""

    original_column_set = set(original_columns)

    missing_columns: List[str] = []
    dtype_issues: List[str] = []
    for column in expected_columns:
        if column not in sensor_data.columns:
            if column in original_column_set:
                dtype_issues.append(column)
            else:
                missing_columns.append(column)
            continue

        converted = pd.to_numeric(sensor_data[column], errors="coerce")
        non_convertible = sensor_data[column].notna() & converted.isna()
        if non_convertible.any():
            dtype_issues.append(column)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise SchemaMismatchError(
            f"Input data is missing required features: {formatted}."
        )

    if dtype_issues:
        formatted = ", ".join(sorted(set(dtype_issues)))
        raise DataTypeMismatchError(
            f"The following features contain non-numeric values: {formatted}."
        )


def _build_event_sensor_payload(
    event_id: int,
    event_data: pd.DataFrame,
    predicted_anomalies: pd.DataFrame,
    arcana_mean_importances: Optional[pd.Series],
    arcana_losses: Optional[pd.DataFrame],
) -> EventSensorData:
    """Create the response payload for a single anomaly event."""

    aligned_anomalies = predicted_anomalies.reindex(event_data.index)

    points: List[SensorPoint] = []
    for (timestamp, sensor_row), (_, anomaly_row) in zip(event_data.iterrows(), aligned_anomalies.iterrows()):
        sensors = {column: _to_optional_float(value) for column, value in sensor_row.items()}
        anomaly_score = anomaly_row.get("anamoly_score")
        threshold_value = anomaly_row.get("threshold_score")
        cumulative_score = anomaly_row.get("cumulative_anamoly_score")

        point = SensorPoint(
            timestamp=_ensure_datetime(timestamp),
            anomaly_score=_to_optional_float(anomaly_score),
            threshold_score=_to_optional_float(threshold_value),
            behaviour=str(anomaly_row.get("behaviour")) if anomaly_row.get("behaviour") is not None else None,
            cumulative_anomaly_score=int(cumulative_score) if pd.notna(cumulative_score) else None,
            sensors=sensors,
        )
        points.append(point)

    mean_importances = (
        {feature: float(value) for feature, value in arcana_mean_importances.items()}
        if arcana_mean_importances is not None
        else {}
    )

    if arcana_losses is not None:
        losses_payload = (
            arcana_losses.reset_index()
            .rename(columns={arcana_losses.index.name or "index": "iteration"})
            .to_dict(orient="records")
        )
    else:
        losses_payload = None

    return EventSensorData(
        event_id=event_id,
        points=points,
        arcana_mean_importances=mean_importances,
        arcana_losses=losses_payload,
    )


def run_prediction(
    request: PredictionRequest,
    detector_loader: Optional[Callable[[str], FaultDetector]] = None,
    events_factory: Optional[
        Callable[[pd.DataFrame, pd.Series, int], Tuple[pd.DataFrame, List[pd.DataFrame]]]
    ] = None,
    event_analyzer: Optional[
        Callable[[FaultDetector, pd.DataFrame, bool], Tuple[pd.Series, pd.DataFrame]]
    ] = None,
) -> PredictionSuccessResponse:
    """Execute a prediction run and build the response payload."""

    if not request.data:
        raise EmptyInputError("No prediction data provided.")

    timestamp_column = request.timestamp_column or DEFAULT_TIMESTAMP_COLUMN

    data_frame = pd.DataFrame(request.data)
    if data_frame.empty:
        raise EmptyInputError("No prediction data provided.")

    if timestamp_column not in data_frame.columns:
        raise TimestampValidationError(
            f"Timestamp column '{timestamp_column}' is missing from the input data."
        )

    try:
        timestamp_index = pd.to_datetime(data_frame[timestamp_column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise TimestampValidationError(
            f"Failed to parse timestamp column '{timestamp_column}': {exc}"
        ) from exc

    feature_data = data_frame.drop(columns=[timestamp_column])
    feature_data.index = timestamp_index
    original_columns = list(feature_data.columns)

    sensor_data = get_sensor_data(feature_data.copy())
    if sensor_data.empty:
        raise DataTypeMismatchError(
            "None of the provided features could be converted to numeric sensor values."
        )

    loader = detector_loader or _load_fault_detector
    detector = loader(request.model_path)

    expected_columns = _extract_expected_columns(detector.data_preprocessor)
    if expected_columns:
        _validate_input_schema(sensor_data, expected_columns, original_columns)
        sensor_data = sensor_data[expected_columns]
    else:
        expected_columns = list(sensor_data.columns)

    sensor_data = sensor_data.sort_index()

    prediction_results: FaultDetectionResult = detector.predict(
        sensor_data=sensor_data, root_cause_analysis=True
    )
    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    anomalies = predicted_anomalies["anomaly"]

    event_creator = events_factory or create_events
    event_meta_data, event_data_list = event_creator(
        sensor_data=sensor_data,
        boolean_information=anomalies,
        min_event_length=request.min_event_length,
    )

    event_ids = list(range(1, len(event_data_list) + 1))

    predicted_anomalies["event_id"] = pd.Series(
        pd.NA, index=predicted_anomalies.index, dtype="Int64"
    )
    predicted_anomalies["critical_event"] = False
    for event_id, event_data in zip(event_ids, event_data_list):
        event_index = event_data.index
        predicted_anomalies.loc[event_index, "event_id"] = event_id
        predicted_anomalies.loc[event_index, "critical_event"] = True

    analyzer = event_analyzer or analyze_event
    event_sensor_payload: List[EventSensorData] = []
    for event_id, event_data in zip(event_ids, event_data_list):
        arcana_mean_importances, arcana_losses = analyzer(
            anomaly_detector=detector,
            event_data=event_data,
            track_losses=False,
        )
        event_sensor_payload.append(
            _build_event_sensor_payload(
                event_id=event_id,
                event_data=event_data,
                predicted_anomalies=predicted_anomalies,
                arcana_mean_importances=arcana_mean_importances,
                arcana_losses=arcana_losses if arcana_losses is not None and not arcana_losses.empty else None,
            )
        )

    events_payload: List[EventMetadata] = []
    for event_id, (_, meta_row) in zip(event_ids, event_meta_data.iterrows()):
        duration = meta_row["duration"]
        if isinstance(duration, pd.Timedelta):
            duration_seconds = float(duration.total_seconds())
        else:
            duration_seconds = float(duration)
        events_payload.append(
            EventMetadata(
                event_id=event_id,
                start=_ensure_datetime(meta_row["start"]),
                end=_ensure_datetime(meta_row["end"]),
                duration_seconds=duration_seconds,
            )
        )

    logger.info(
        "Prediction finished for model '%s'. Detected %d anomaly events.",
        request.model_path,
        len(events_payload),
    )

    return PredictionSuccessResponse(
        status="success",
        events=events_payload,
        event_sensor_data=event_sensor_payload,
    )


app = FastAPI(title="Energy Fault Detector Prediction API")


@app.post(
    "/predict",
    response_model=PredictionSuccessResponse,
    responses={
        404: {"description": "Model not found."},
        409: {"description": "Schema mismatch."},
        422: {"description": "Data validation error."},
        500: {"description": "Internal server error."},
    },
)
async def predict(request: PredictionRequest) -> PredictionSuccessResponse:
    """Predict anomalies for the provided payload."""

    try:
        return run_prediction(request)
    except PredictionAPIError as exc:
        logger.exception("Prediction request failed: %s", exc.message)
        raise HTTPException(
            status_code=exc.status_code,
            detail={"status": "error", "code": exc.error_code, "message": exc.message},
        ) from exc
    except Exception as exc:  # pragma: no cover - safeguard for unexpected failures
        logger.exception("Unexpected error during prediction request handling.")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "code": "EFD_INTERNAL_ERROR",
                "message": "Unexpected internal server error.",
            },
        ) from exc


__all__ = [
    "app",
    "run_prediction",
    "PredictionRequest",
    "PredictionSuccessResponse",
    "PredictionAPIError",
    "SchemaMismatchError",
    "DataTypeMismatchError",
    "InvalidModelError",
    "TimestampValidationError",
    "EmptyInputError",
]
