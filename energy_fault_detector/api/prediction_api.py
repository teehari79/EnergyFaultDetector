"""FastAPI application exposing the quick fault detector prediction endpoint."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from energy_fault_detector._logs import setup_logging
from energy_fault_detector.config import Config
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.quick_fault_detection.data_loading import get_sensor_data
from energy_fault_detector.quick_fault_detection.quick_fault_detector import analyze_event
from energy_fault_detector.utils.analysis import create_events
from energy_fault_detector.api.model_registry import ModelNotFoundError, ModelRegistry
from energy_fault_detector.api.settings import get_settings

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


class JobNotFoundError(PredictionAPIError):
    """Raised when a requested asynchronous job cannot be found."""

    status_code = 404
    error_code = "EFD_JOB_NOT_FOUND"


class AuthenticationError(PredictionAPIError):
    """Raised when a client fails to authenticate correctly."""

    status_code = 401
    error_code = "EFD_AUTH_FAILED"


class AuthorizationError(PredictionAPIError):
    """Raised when an authenticated client is not authorised for an action."""

    status_code = 403
    error_code = "EFD_AUTHORIZATION_FAILED"


class DecryptionError(PredictionAPIError):
    """Raised when encrypted content cannot be decrypted."""

    status_code = 403
    error_code = "EFD_DECRYPTION_FAILED"


class PayloadValidationError(PredictionAPIError):
    """Raised when the decrypted payload fails validation."""

    status_code = 422
    error_code = "EFD_PAYLOAD_INVALID"


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


def _http_exception_from_error(exc: PredictionAPIError) -> HTTPException:
    """Convert a domain specific error into an :class:`HTTPException`."""

    return HTTPException(
        status_code=exc.status_code,
        detail={"status": "error", "code": exc.error_code, "message": exc.message},
    )


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

    status: Literal["success"] = Field(
        default="success",
        description="Indicates the prediction request completed successfully.",
    )
    events: List[EventMetadata]
    event_sensor_data: List[EventSensorData]


@dataclass
class AuthSession:
    """Authenticated session associated with an organisation."""

    auth_token: str
    organization_id: str
    username: str
    seed_token: str
    expires_at: datetime


@dataclass
class PredictionJobRecord:
    """Represents the asynchronous processing state for a prediction request."""

    auth_token: str
    organization_id: str
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    result_path: Optional[str] = None
    error: Optional[str] = None
    webhooks: Dict[str, str] = field(default_factory=dict)


_AUTH_SESSIONS: Dict[str, AuthSession] = {}
_AUTH_LOCK = asyncio.Lock()
_PREDICTION_JOBS: Dict[str, PredictionJobRecord] = {}
_JOBS_LOCK = asyncio.Lock()


class PredictionRequest(BaseModel):
    """Payload describing a prediction request."""

    model_path: str = Field(..., description="Path to the saved fault detector model.")
    data: List[Dict[str, Any]] = Field(..., min_items=1, description="Prediction data as list of records.")
    timestamp_column: str = Field(
        DEFAULT_TIMESTAMP_COLUMN,
        description="Name of the timestamp column in the provided data.",
    )
    min_event_length: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Minimum number of consecutive anomalies required to flag a critical event."
            " Set to 0 to disable the length requirement."
        ),
    )
    min_event_duration: Optional[Union[str, float, int]] = Field(
        None,
        description=(
            "Minimum duration that anomalies must span to be considered critical."
            " Strings follow pandas timedelta notation while numeric values are interpreted as seconds."
        ),
    )
    enable_narrative: Optional[bool] = Field(
        None,
        description=(
            "When explicitly set to False the asynchronous pipeline skips NLP narrative generation."
        ),
    )


class FilePredictionRequest(BaseModel):
    """Prediction request referencing artefacts managed by the API service."""

    model_name: str = Field(..., description="Logical model identifier managed by the service.")
    data_path: str = Field(..., description="Filesystem path to the CSV file containing prediction data.")
    model_version: Optional[str] = Field(
        None,
        description=(
            "Specific model version to use. Defaults to the latest available version registered for the model."
        ),
    )
    timestamp_column: str = Field(
        DEFAULT_TIMESTAMP_COLUMN,
        description="Name of the timestamp column present in the CSV input file.",
    )
    min_event_length: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Minimum number of consecutive anomalies required to flag a critical event."
            " Set to 0 to disable the length requirement."
        ),
    )
    min_event_duration: Optional[Union[str, float, int]] = Field(
        None,
        description=(
            "Minimum duration that anomalies must span to be considered critical."
            " Strings follow pandas timedelta notation while numeric values are interpreted as seconds."
        ),
    )
    enable_narrative: Optional[bool] = Field(
        None,
        description=(
            "When explicitly set to False the asynchronous pipeline skips NLP narrative generation."
        ),
    )
    asset_name: Optional[str] = Field(
        None,
        description=(
            "Optional asset identifier forwarded for informational purposes. Currently unused by the async API."
        ),
    )
    ignore_features: List[str] = Field(
        default_factory=list,
        description=(
            "Feature names or wildcard patterns to exclude during post-processing."
            " Accepted for compatibility with synchronous API requests."
        ),
    )
    debug_plots: Optional[bool] = Field(
        None,
        description="Optional flag mirroring the synchronous API behaviour. Currently unused by the async API.",
    )


@lru_cache()
def _get_model_registry() -> ModelRegistry:
    """Return a cached :class:`ModelRegistry` instance backed by the service settings."""

    settings = get_settings()
    return ModelRegistry(
        root_directory=settings.model_store.root_directory,
        default_version_strategy=settings.model_store.default_version_strategy,
    )


def _resolve_model_path(model_name: str, model_version: Optional[str]) -> str:
    """Resolve ``model_name``/``model_version`` to a filesystem path."""

    registry = _get_model_registry()
    try:
        model_path, _ = registry.resolve(model_name, model_version)
    except ModelNotFoundError as exc:
        raise InvalidModelError(str(exc)) from exc
    return str(model_path)


def _load_file_prediction_data(data_path: str) -> List[Dict[str, Any]]:
    """Load prediction data from ``data_path`` and return it as a list of records."""

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise PayloadValidationError(f"Prediction data file '{path}' does not exist.")

    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:  # pragma: no cover - handled by explicit check above
        raise PayloadValidationError(f"Prediction data file '{path}' does not exist.") from exc
    except pd.errors.EmptyDataError as exc:
        raise PayloadValidationError(f"Prediction data file '{path}' is empty.") from exc
    except Exception as exc:  # pragma: no cover - pandas raises various subclasses depending on contents
        raise PayloadValidationError(
            f"Failed to read prediction data from '{path}': {exc}"
        ) from exc

    if frame.empty:
        raise PayloadValidationError(f"Prediction data file '{path}' does not contain any rows.")

    return frame.to_dict(orient="records")


def _prepare_prediction_request(
    request: Union[PredictionRequest, FilePredictionRequest]
) -> PredictionRequest:
    """Normalise external prediction requests to :class:`PredictionRequest`."""

    if isinstance(request, PredictionRequest):
        return request

    model_path = _resolve_model_path(request.model_name, request.model_version)
    records = _load_file_prediction_data(request.data_path)
    timestamp_column = request.timestamp_column or DEFAULT_TIMESTAMP_COLUMN

    return PredictionRequest(
        model_path=model_path,
        data=records,
        timestamp_column=timestamp_column,
        min_event_length=request.min_event_length,
        min_event_duration=request.min_event_duration,
        enable_narrative=request.enable_narrative,
    )


def _derive_key(seed_token: str, context: str) -> bytes:
    """Derive a symmetric key using the tenant seed token and provided context."""

    material = f"{seed_token}:{context}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _xor_cipher(data: bytes, key: bytes) -> bytes:
    """Apply a simple XOR cipher using the provided key."""

    return bytes(byte ^ key[idx % len(key)] for idx, byte in enumerate(data))


def _decrypt_payload(seed_token: str, payload: str, context: str) -> str:
    """Decrypt a base64 encoded payload using the tenant seed token and context."""

    try:
        encrypted = base64.b64decode(payload.encode("utf-8"))
    except (ValueError, binascii.Error) as exc:
        raise DecryptionError("Encrypted payload is not valid base64 data.") from exc

    key = _derive_key(seed_token, context)
    decrypted = _xor_cipher(encrypted, key)

    try:
        return decrypted.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecryptionError("Decrypted payload contains invalid text.") from exc


def _encrypt_payload(seed_token: str, content: Dict[str, Any], context: str) -> str:
    """Encrypt ``content`` using the same mechanism as the caller for testing purposes."""

    raw = json.dumps(content).encode("utf-8")
    key = _derive_key(seed_token, context)
    encrypted = _xor_cipher(raw, key)
    return base64.b64encode(encrypted).decode("utf-8")


def _hash_auth_token(auth_token: str, seed_token: str) -> str:
    """Hash the auth token and tenant seed token to produce the encryption context."""

    digest = hashlib.sha256(f"{auth_token}:{seed_token}".encode("utf-8")).hexdigest()
    return digest


def _verify_password(password: str, stored_hash: str) -> bool:
    """Validate a password against a stored PBKDF2-SHA256 hash."""

    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$")
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False

    try:
        iteration_count = int(iterations)
    except ValueError:
        return False

    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iteration_count)
    return hmac.compare_digest(computed.hex(), digest_hex)


def _utcnow() -> datetime:
    """Return a timezone aware UTC ``datetime``."""

    return datetime.now(timezone.utc)


async def _store_session(session: AuthSession) -> None:
    async with _AUTH_LOCK:
        _AUTH_SESSIONS[session.auth_token] = session


async def _get_session(auth_token: str) -> AuthSession:
    async with _AUTH_LOCK:
        session = _AUTH_SESSIONS.get(auth_token)

    if session is None:
        raise AuthenticationError("Invalid authentication token provided.")

    if session.expires_at <= _utcnow():
        async with _AUTH_LOCK:
            _AUTH_SESSIONS.pop(auth_token, None)
        raise AuthenticationError("Authentication token has expired. Please re-authenticate.")

    return session


async def _create_job(auth_token: str, organization_id: str, webhooks: Dict[str, str]) -> str:
    job_id = str(uuid4())
    record = PredictionJobRecord(
        auth_token=auth_token,
        organization_id=organization_id,
        status="pending",
        created_at=_utcnow(),
        updated_at=_utcnow(),
        webhooks=webhooks,
    )
    async with _JOBS_LOCK:
        _PREDICTION_JOBS[job_id] = record
    return job_id


async def _get_job(job_id: str) -> PredictionJobRecord:
    async with _JOBS_LOCK:
        record = _PREDICTION_JOBS.get(job_id)
    if record is None:
        raise JobNotFoundError(f"Prediction job '{job_id}' could not be found.")
    return record


async def _update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    step: Optional[str] = None,
    step_output: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    result_path: Optional[str] = None,
) -> None:
    async with _JOBS_LOCK:
        record = _PREDICTION_JOBS.get(job_id)
        if record is None:
            return

        if status is not None:
            record.status = status
        if step is not None and step_output is not None:
            record.step_outputs[step] = step_output
        if error is not None:
            record.error = error
        if result_path is not None:
            record.result_path = result_path
        record.updated_at = _utcnow()


def _get_tenant_security(organization_id: str):
    settings = get_settings()
    tenant = settings.security.tenants.get(organization_id)
    if tenant is None:
        raise AuthenticationError(
            f"Unknown organisation identifier '{organization_id}'. Please verify the organisation id."
        )
    return tenant, settings.security


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


def _load_sensor_descriptions(model_path: Optional[str]) -> Dict[str, str]:
    """Load sensor name to description mapping from the model directory, if available."""

    if not model_path:
        return {}

    sensor_file = Path(model_path) / "sensor_data.csv"
    if not sensor_file.exists():
        return {}

    try:
        sensor_df = pd.read_csv(sensor_file)
    except (OSError, ValueError, pd.errors.EmptyDataError) as exc:  # pragma: no cover - filesystem dependent
        logger.warning("Failed to read sensor description file at '%s': %s", sensor_file, exc)
        return {}

    lower_columns = {column.strip().lower(): column for column in sensor_df.columns}
    sensor_column = lower_columns.get("sensor name")
    description_column = lower_columns.get("feature description")
    if not sensor_column or not description_column:
        logger.info(
            "Sensor description file at '%s' does not contain required columns 'sensor name' and 'feature description'.",
            sensor_file,
        )
        return {}

    sensor_descriptions: Dict[str, str] = {}
    for _, row in sensor_df[[sensor_column, description_column]].dropna().iterrows():
        sensor_name = str(row[sensor_column]).strip()
        feature_description = str(row[description_column]).strip()
        if sensor_name and feature_description:
            sensor_descriptions[sensor_name] = feature_description

    return sensor_descriptions


def _describe_sensor_feature(feature_name: str, sensor_name_mapping: Mapping[str, str]) -> str:
    """Return a human readable name for a sensor feature based on the provided mapping."""

    if not sensor_name_mapping:
        return feature_name

    if feature_name in sensor_name_mapping:
        return sensor_name_mapping[feature_name]

    parts = [part for part in re.split(r"[_\s]+", feature_name) if part]
    if not parts:
        return feature_name

    for end_idx in range(len(parts), 0, -1):
        prefix = "_".join(parts[:end_idx])
        description = sensor_name_mapping.get(prefix)
        if description:
            suffix = "_".join(parts[end_idx:])
            if suffix:
                formatted_suffix = suffix.replace("_", " ")
                return f"{description} ({formatted_suffix})"
            return description

    return feature_name


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
    sensor_name_mapping: Mapping[str, str],
) -> EventSensorData:
    """Create the response payload for a single anomaly event."""

    aligned_anomalies = predicted_anomalies.reindex(event_data.index)

    points: List[SensorPoint] = []
    for (timestamp, sensor_row), (_, anomaly_row) in zip(event_data.iterrows(), aligned_anomalies.iterrows()):
        sensors = {
            _describe_sensor_feature(column, sensor_name_mapping): _to_optional_float(value)
            for column, value in sensor_row.items()
        }
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

    mean_importances: Dict[str, float] = {}
    if arcana_mean_importances is not None:
        mean_importances = {
            _describe_sensor_feature(feature, sensor_name_mapping): float(value)
            for feature, value in arcana_mean_importances.items()
        }

    if arcana_losses is not None:
        renamed_losses = arcana_losses.rename(
            columns=lambda column: _describe_sensor_feature(column, sensor_name_mapping)
        )
        losses_payload = (
            renamed_losses.reset_index()
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
        Callable[[pd.DataFrame, pd.Series, Optional[int], Optional[Union[str, float, int]]],
                 Tuple[pd.DataFrame, List[pd.DataFrame]]]
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

    sensor_name_mapping = _load_sensor_descriptions(request.model_path)

    prediction_results: FaultDetectionResult = detector.predict(
        sensor_data=sensor_data, root_cause_analysis=True
    )
    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    anomalies = predicted_anomalies["anomaly"]

    config_min_length = getattr(detector.config, "critical_event_min_length", None) if detector.config else None
    config_min_duration = getattr(detector.config, "critical_event_min_duration", None) if detector.config else None

    effective_min_length: Optional[int]
    if request.min_event_length is not None:
        effective_min_length = request.min_event_length
    elif config_min_length is not None:
        effective_min_length = config_min_length
    else:
        effective_min_length = DEFAULT_MIN_EVENT_LENGTH

    if effective_min_length is not None and effective_min_length < 1:
        effective_min_length = None

    effective_min_duration = (
        request.min_event_duration if request.min_event_duration is not None else config_min_duration
    )

    event_creator = events_factory or create_events
    event_meta_data, event_data_list = event_creator(
        sensor_data=sensor_data,
        boolean_information=anomalies,
        min_event_length=effective_min_length,
        min_event_duration=effective_min_duration,
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
                sensor_name_mapping=sensor_name_mapping,
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


def _normalise_duration_seconds(value: Optional[Union[str, float, int]]) -> Optional[float]:
    """Normalise event duration thresholds to seconds."""

    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        duration = pd.to_timedelta(value)
    except (ValueError, TypeError) as exc:
        raise PayloadValidationError(
            f"Invalid duration specification '{value}'."
        ) from exc
    return float(duration.total_seconds())


def _summarise_anomalies(result: PredictionSuccessResponse) -> Dict[str, Any]:
    total_points = sum(len(event.points) for event in result.event_sensor_data)
    return {
        "total_events": len(result.events),
        "total_points": total_points,
        "event_ids": [event.event_id for event in result.events],
    }


def _serialise_events(result: PredictionSuccessResponse) -> List[Dict[str, Any]]:
    return [event.dict() for event in result.events]


def _serialise_event_details(result: PredictionSuccessResponse) -> List[Dict[str, Any]]:
    return [event_data.dict() for event_data in result.event_sensor_data]


def _identify_critical_events(
    result: PredictionSuccessResponse,
    min_event_length: Optional[int],
    min_event_duration: Optional[Union[str, float, int]],
) -> List[Dict[str, Any]]:
    duration_threshold = None
    if min_event_duration not in (None, 0):
        duration_threshold = _normalise_duration_seconds(min_event_duration)

    critical_events: List[Dict[str, Any]] = []
    for event, sensor_data in zip(result.events, result.event_sensor_data):
        length_ok = True
        if min_event_length not in (None, 0):
            length_ok = len(sensor_data.points) >= int(min_event_length)

        duration_ok = True
        if duration_threshold is not None:
            duration_ok = event.duration_seconds >= duration_threshold

        if length_ok and duration_ok:
            payload = event.dict()
            payload["sample_count"] = len(sensor_data.points)
            critical_events.append(payload)

    return critical_events


def _compile_root_cause(result: PredictionSuccessResponse) -> List[Dict[str, Any]]:
    analyses: List[Dict[str, Any]] = []
    for event, sensor_data in zip(result.events, result.event_sensor_data):
        ranked = sorted(
            sensor_data.arcana_mean_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        analyses.append({"event_id": event.event_id, "ranked_sensors": ranked})
    return analyses


def _build_event_narratives(result: PredictionSuccessResponse) -> List[Dict[str, Any]]:
    narratives: List[Dict[str, Any]] = []
    for event, sensor_data in zip(result.events, result.event_sensor_data):
        ranked = sorted(
            sensor_data.arcana_mean_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            top_description = ", ".join(f"{name} ({score:.2f})" for name, score in ranked[:3])
        else:
            top_description = "no dominant sensors"

        narratives.append(
            {
                "event_id": event.event_id,
                "narrative": (
                    f"Between {event.start.isoformat()} and {event.end.isoformat()} the model detected "
                    f"an abnormal pattern primarily driven by {top_description}."
                ),
            }
        )
    return narratives


def _job_artifact_directory(job_id: str) -> Path:
    """Return the filesystem directory where artifacts for ``job_id`` are stored."""

    settings = get_settings()
    base_dir = settings.prediction.output_directory
    if base_dir is None:
        base_dir = Path.cwd() / "artifacts"
    job_dir = (base_dir / "async_jobs" / job_id).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _persist_json_artifact(job_id: str, name: str, data: Any) -> str:
    """Serialise ``data`` to disk and return the created file path."""

    job_dir = _job_artifact_directory(job_id)
    file_path = job_dir / f"{name}.json"
    serialisable = jsonable_encoder(data)
    with file_path.open("w", encoding="utf-8") as stream:
        json.dump(serialisable, stream, ensure_ascii=False, indent=2)
    return str(file_path)


async def _post_webhook(url: str, payload: Dict[str, Any], auth_token: str, job_id: str) -> Dict[str, Any]:
    headers = {
        "X-EFD-Auth-Token": auth_token,
        "X-EFD-Job-ID": job_id,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        status = "delivered"
        error_message = None
    except httpx.HTTPError as exc:  # pragma: no cover - network errors depend on environment
        logger.warning("Webhook delivery to %s failed: %s", url, exc)
        status = "failed"
        error_message = str(exc)
    return {"url": url, "status": status, "error": error_message}


async def _notify_step(
    job_id: str,
    step: str,
    data: Dict[str, Any],
    webhook_url: Optional[str],
    auth_token: str,
) -> Dict[str, Any]:
    payload = {"step": step, "data": jsonable_encoder(data)}
    data_path = _persist_json_artifact(job_id, step, data)
    delivery: Optional[Dict[str, Any]] = None
    if webhook_url:
        delivery = await _post_webhook(webhook_url, payload, auth_token, job_id)
    result = {"data_path": data_path}
    if delivery:
        result["webhook"] = delivery
    return result


class PredictionWebhooks(BaseModel):
    """Webhook configuration for asynchronous pipeline notifications."""

    model_config = ConfigDict(populate_by_name=True)

    anomalies: Optional[HttpUrl] = Field(
        None, description="Webhook invoked after anomaly detection summaries are prepared."
    )
    events: Optional[HttpUrl] = Field(
        None, description="Webhook invoked once event metadata is available."
    )
    criticality: Optional[HttpUrl] = Field(
        None,
        description="Webhook invoked once critical events have been identified.",
        alias="critical",
    )
    root_cause: Optional[HttpUrl] = Field(
        None, description="Webhook invoked once root cause scores are derived."
    )
    narrative: Optional[HttpUrl] = Field(
        None, description="Webhook invoked after the narrative summary is generated."
    )

class AsyncPredictionPayload(BaseModel):
    """Decrypted payload describing an asynchronous prediction job."""

    organization_id: str = Field(..., description="Identifier of the calling organisation.")
    request: Union[PredictionRequest, FilePredictionRequest] = Field(
        ..., description="Actual prediction request payload."
    )
    webhooks: Optional[PredictionWebhooks] = Field(
        None, description="Optional webhook endpoints that receive pipeline updates."
    )


class EncryptedPredictionRequest(BaseModel):
    """Request body for the asynchronous prediction endpoint."""

    auth_token: str = Field(..., description="Authentication token issued by the auth endpoint.")
    auth_hash: str = Field(
        ...,
        description=(
            "Hash derived from the auth token and tenant seed. Also used as encryption context for the payload."
        ),
    )
    payload_encrypted: str = Field(..., description="Encrypted payload containing the prediction request.")


class AuthenticationRequest(BaseModel):
    """Authentication request carrying encrypted credentials."""

    organization_id: str = Field(..., description="Identifier of the calling organisation.")
    credentials_encrypted: str = Field(
        ..., description="Username and password encrypted with the shared seed token."
    )
    nonce: Optional[str] = Field(
        None,
        description=(
            "Optional nonce that callers may include for additional replay protection. Currently informational."
        ),
    )


class AuthenticationResponse(BaseModel):
    """Response returned after successful authentication."""

    status: Literal["authenticated"] = Field(
        default="authenticated",
        description="Indicates the authentication request completed successfully.",
    )
    auth_token: str = Field(..., description="Authentication token to be used for subsequent calls.")
    expires_at: datetime = Field(..., description="UTC timestamp when the token expires.")


class AsyncPredictionResponse(BaseModel):
    """Response acknowledging acceptance of an asynchronous prediction job."""

    status: Literal["accepted"] = Field(
        default="accepted",
        description="Indicates the asynchronous prediction job was accepted for processing.",
    )
    job_id: str = Field(..., description="Identifier of the asynchronous prediction job.")


class StepArtifact(BaseModel):
    """Reference to a persisted pipeline artefact."""

    data_path: str = Field(..., description="Filesystem path to the persisted artefact.")
    webhook: Optional[Dict[str, Any]] = Field(
        None, description="Result of delivering the artefact payload to the webhook, if configured."
    )


class JobStatusResponse(BaseModel):
    """Status payload returned when querying a prediction job."""

    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    steps: Dict[str, StepArtifact] = Field(default_factory=dict)
    result_path: Optional[str] = Field(
        None,
        description="Filesystem path to the final prediction result payload when the job completes.",
    )
    error: Optional[str] = None


async def _run_async_prediction_job(
    job_id: str,
    payload: AsyncPredictionPayload,
    session: AuthSession,
) -> None:
    webhook_urls: Dict[str, str] = {}
    if payload.webhooks is not None:
        raw_webhooks = payload.webhooks.dict(exclude_none=True, by_alias=False)
        webhook_urls = {key: str(value) for key, value in raw_webhooks.items()}

    try:
        await _update_job(job_id, status="running")
        logger.info(
            "Job %s started for organisation '%s' (auth token %s).",
            job_id,
            session.organization_id,
            session.auth_token,
        )

        prediction_request = _prepare_prediction_request(payload.request)
        narrative_enabled = prediction_request.enable_narrative is not False

        prediction_result = await asyncio.to_thread(run_prediction, prediction_request)

        anomalies_summary = _summarise_anomalies(prediction_result)
        anomalies_output = await _notify_step(
            job_id,
            "anomaly_detection",
            anomalies_summary,
            webhook_urls.get("anomalies"),
            session.auth_token,
        )
        await _update_job(job_id, step="anomaly_detection", step_output=anomalies_output)

        events_payload = {
            "events": _serialise_events(prediction_result),
            "event_sensor_data": _serialise_event_details(prediction_result),
        }
        events_output = await _notify_step(
            job_id,
            "event_detection",
            events_payload,
            webhook_urls.get("events"),
            session.auth_token,
        )
        await _update_job(job_id, step="event_detection", step_output=events_output)

        critical_events = _identify_critical_events(
            prediction_result,
            prediction_request.min_event_length,
            prediction_request.min_event_duration,
        )
        critical_payload = {"events": critical_events}
        critical_output = await _notify_step(
            job_id,
            "criticality_analysis",
            critical_payload,
            webhook_urls.get("criticality"),
            session.auth_token,
        )
        await _update_job(job_id, step="criticality_analysis", step_output=critical_output)

        root_cause_payload = {"root_cause_analysis": _compile_root_cause(prediction_result)}
        root_cause_output = await _notify_step(
            job_id,
            "root_cause_analysis",
            root_cause_payload,
            webhook_urls.get("root_cause"),
            session.auth_token,
        )
        await _update_job(job_id, step="root_cause_analysis", step_output=root_cause_output)

        if narrative_enabled:
            narratives_payload = {"narratives": _build_event_narratives(prediction_result)}
            narratives_output = await _notify_step(
                job_id,
                "narrative_generation",
                narratives_payload,
                webhook_urls.get("narrative"),
                session.auth_token,
            )
            await _update_job(
                job_id, step="narrative_generation", step_output=narratives_output
            )
        else:
            logger.info("Narrative generation disabled for job %s", job_id)

        result_path = _persist_json_artifact(job_id, "prediction_result", prediction_result)
        await _update_job(job_id, status="completed", result_path=result_path)
        logger.info(
            "Job %s completed for organisation '%s'.", job_id, session.organization_id
        )
    except PredictionAPIError as exc:
        logger.exception("Job %s failed due to handled error: %s", job_id, exc.message)
        await _update_job(job_id, status="failed", error=exc.message)
    except Exception as exc:  # pragma: no cover - safeguard for unexpected failures
        logger.exception("Job %s failed due to unexpected error.", job_id)
        await _update_job(job_id, status="failed", error=str(exc))


app = FastAPI(title="Energy Fault Detector Prediction API")


@app.post(
    "/auth",
    response_model=AuthenticationResponse,
    responses={
        401: {"description": "Authentication failed."},
        403: {"description": "Decryption failed."},
    },
)
async def authenticate(request: AuthenticationRequest) -> AuthenticationResponse:
    """Authenticate a caller and return an auth token for subsequent requests."""

    try:
        tenant, security_settings = _get_tenant_security(request.organization_id)
        decrypted = _decrypt_payload(tenant.seed_token, request.credentials_encrypted, "auth_credentials")
        try:
            credentials = json.loads(decrypted)
        except json.JSONDecodeError as exc:
            raise DecryptionError("Decrypted credentials were not valid JSON.") from exc

        username = credentials.get("username")
        password = credentials.get("password")
        if not username or not password:
            raise AuthenticationError("Both username and password must be provided in the encrypted payload.")

        stored_hash = tenant.users.get(str(username))
        if stored_hash is None or not _verify_password(str(password), stored_hash):
            raise AuthenticationError("Invalid username or password provided.")

        expires_at = _utcnow() + timedelta(seconds=security_settings.token_ttl_seconds)
        auth_token = str(uuid4())
        session = AuthSession(
            auth_token=auth_token,
            organization_id=request.organization_id,
            username=str(username),
            seed_token=tenant.seed_token,
            expires_at=expires_at,
        )
        await _store_session(session)

        logger.info(
            "Issued auth token %s for organisation '%s' and user '%s'.",
            auth_token,
            request.organization_id,
            username,
        )

        return AuthenticationResponse(auth_token=auth_token, expires_at=expires_at)
    except PredictionAPIError as exc:
        logger.warning(
            "Authentication request failed for organisation '%s': %s",
            request.organization_id,
            exc.message,
        )
        raise _http_exception_from_error(exc)


@app.post(
    "/predict",
    response_model=AsyncPredictionResponse,
    responses={
        401: {"description": "Authentication required."},
        403: {"description": "Hash mismatch or payload decryption failed."},
        422: {"description": "Prediction payload validation failed."},
        500: {"description": "Internal server error."},
    },
)
async def predict(request: EncryptedPredictionRequest) -> AsyncPredictionResponse:
    """Accept an encrypted prediction payload and start asynchronous processing."""

    try:
        session = await _get_session(request.auth_token)
        provided_hash = request.auth_hash.strip().lower()
        expected_hash = _hash_auth_token(request.auth_token, session.seed_token)
        if not hmac.compare_digest(expected_hash, provided_hash):
            raise DecryptionError("Authentication hash mismatch.")

        decrypted_payload = _decrypt_payload(session.seed_token, request.payload_encrypted, provided_hash)
        try:
            payload_dict = json.loads(decrypted_payload)
        except json.JSONDecodeError as exc:
            raise DecryptionError("Decrypted payload was not valid JSON.") from exc

        try:
            payload = AsyncPredictionPayload(**payload_dict)
        except ValidationError as exc:
            raise PayloadValidationError("Invalid prediction payload provided.") from exc

        if payload.organization_id != session.organization_id:
            raise AuthorizationError(
                "Organisation identifier in payload does not match the authenticated session."
            )

        webhook_urls = (
            payload.webhooks.dict(exclude_none=True, by_alias=False)
            if payload.webhooks is not None
            else {}
        )
        job_id = await _create_job(
            auth_token=session.auth_token,
            organization_id=session.organization_id,
            webhooks={key: str(value) for key, value in webhook_urls.items()},
        )

        asyncio.create_task(_run_async_prediction_job(job_id, payload, session))

        logger.info(
            "Accepted prediction job %s for organisation '%s'.",
            job_id,
            session.organization_id,
        )

        return AsyncPredictionResponse(job_id=job_id)
    except PredictionAPIError as exc:
        logger.warning(
            "Prediction request rejected for token '%s': %s",
            request.auth_token,
            exc.message,
        )
        raise _http_exception_from_error(exc)


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={
        401: {"description": "Authentication required."},
        403: {"description": "Access to the requested job is forbidden."},
        404: {"description": "Job not found."},
    },
)
async def job_status(job_id: str, auth_token: str) -> JobStatusResponse:
    """Retrieve the current status of an asynchronous prediction job."""

    try:
        session = await _get_session(auth_token)
        record = await _get_job(job_id)

        if record.auth_token != session.auth_token:
            raise AuthorizationError("The provided auth token does not grant access to this job.")

        return JobStatusResponse(
            job_id=job_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            steps=record.step_outputs,
            result_path=record.result_path,
            error=record.error,
        )
    except PredictionAPIError as exc:
        logger.warning("Job status request failed: %s", exc.message)
        raise _http_exception_from_error(exc)


__all__ = [
    "app",
    "run_prediction",
    "PredictionRequest",
    "FilePredictionRequest",
    "PredictionSuccessResponse",
    "PredictionAPIError",
    "SchemaMismatchError",
    "DataTypeMismatchError",
    "InvalidModelError",
    "JobNotFoundError",
    "TimestampValidationError",
    "EmptyInputError",
    "AuthenticationRequest",
    "AuthenticationResponse",
    "EncryptedPredictionRequest",
    "AsyncPredictionResponse",
    "StepArtifact",
    "JobStatusResponse",
    "PredictionWebhooks",
    "AuthenticationError",
    "AuthorizationError",
    "DecryptionError",
    "PayloadValidationError",
]
