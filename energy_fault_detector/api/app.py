"""FastAPI application exposing prediction endpoints."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, AsyncGenerator
from uuid import uuid4

from fastapi import File, Form, HTTPException, UploadFile, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import EventSourceResponse, JSONResponse, Response

import pandas as pd
from pydantic import BaseModel, Field, validator

from energy_fault_detector.quick_fault_detection import quick_fault_detector

from .model_registry import ModelNotFoundError, ModelRegistry
from .narrative_pipeline import (
    NarrativeContext,
    NarrativePipeline,
    derive_event_insights,
    gather_root_cause_hypotheses,
    summarise_configuration,
)
from .perplexity import PerplexityClient
from .llm_factory import LLMConfiguration, create_chat_model
from .settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Lightweight container storing derived insights for uploaded datasets."""

    prediction_id: str
    batch_name: str
    notes: Optional[str]
    file_path: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    critical_events: List[Dict[str, Any]] = field(default_factory=list)
    root_causes: List[Dict[str, Any]] = field(default_factory=list)
    narrative: Optional[str] = None
    status: str = "processing"
    error: Optional[str] = None

    def summary_counts(self) -> Dict[str, int]:
        """Return derived counts used for reporting and narratives."""

        return {
            "anomalies": len(self.anomalies),
            "critical": len(self.critical_events),
            "root_causes": len(self.root_causes),
        }


PREDICTION_STORE: Dict[str, PredictionRecord] = {}


def _merge_ignore_patterns(defaults: List[str], overrides: List[str]) -> List[str]:
    merged: List[str] = []
    for values in (defaults or [], overrides or []):
        for value in values:
            if value not in merged:
                merged.append(value)
    return merged


def _dataframe_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    serialisable = df.reset_index()
    serialisable = serialisable.where(pd.notnull(serialisable), None)
    return json.loads(serialisable.to_json(orient="records", date_format="iso"))


def _determine_asset_name(model_name: str, model_version: str, explicit: Optional[str], template: str) -> str:
    if explicit:
        return explicit

    try:
        return template.format(model_name=model_name, model_version=model_version)
    except KeyError as exc:  # pragma: no cover - configuration error
        raise HTTPException(
            status_code=500,
            detail=f"Invalid asset name template; missing placeholder: {exc}",
        ) from exc


def _derive_artifact_directory(save_root: Optional[Path], data_path: Path, asset_name: str) -> Path:
    base_dir = save_root if save_root is not None else data_path.resolve().parent
    return base_dir / "prediction_output" / asset_name


def _execute_prediction_pipeline(
    request: PredictionRequest,
) -> Dict[str, Any]:
    try:
        if request.asset_name:
            model_path, resolved_version = model_registry.resolve_from_asset_name(
                request.asset_name, request.model_version
            )
        else:
            model_path, resolved_version = model_registry.resolve(
                model_name=request.model_name, model_version=request.model_version
            )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    data_path = Path(request.data_path).expanduser().resolve()
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Data file '{data_path}' does not exist.")

    ignore_patterns = _merge_ignore_patterns(
        settings.prediction.default_ignore_features, request.ignore_features
    )
    debug_plots = request.debug_plots if request.debug_plots is not None else settings.prediction.debug_plots

    asset_name = _determine_asset_name(
        model_name=request.model_name,
        model_version=resolved_version,
        explicit=request.asset_name,
        template=settings.prediction.default_asset_name,
    )

    save_root = settings.prediction.output_directory
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)

    try:
        prediction_results, event_metadata, event_analysis, _ = quick_fault_detector(
            csv_data_path=None,
            csv_test_data_path=str(data_path),
            train_test_column_name=settings.prediction.train_test_column,
            train_test_mapping=settings.prediction.train_test_mapping or None,
            time_column_name=settings.prediction.time_column,
            status_data_column_name=settings.prediction.status_column,
            status_mapping=settings.prediction.status_mapping or None,
            min_anomaly_length=settings.prediction.min_anomaly_length,
            critical_event_min_length=settings.prediction.critical_event_min_length,
            critical_event_min_duration=settings.prediction.critical_event_min_duration,
            enable_debug_plots=debug_plots,
            save_dir=str(save_root) if save_root is not None else None,
            mode="predict",
            model_path=str(model_path),
            asset_name=asset_name,
            rca_ignore_features=ignore_patterns or None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive programming
        logger.exception("Prediction request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed. See logs for details.") from exc

    artifact_directory = _derive_artifact_directory(save_root, data_path, asset_name)

    return {
        "prediction_results": prediction_results,
        "event_metadata": event_metadata,
        "event_analysis": event_analysis,
        "resolved_version": resolved_version,
        "asset_name": asset_name,
        "ignore_patterns": ignore_patterns,
        "artifact_directory": artifact_directory,
        "data_path": data_path,
    }


class PredictionRequest(BaseModel):
    """Request body for the prediction endpoint."""

    model_name: str = Field(..., description="Logical name of the model to load.")
    data_path: str = Field(..., description="Path to the CSV file that should be analysed.")
    model_version: Optional[str] = Field(
        None, description="Specific model version to use. Defaults to the latest available version."
    )
    ignore_features: List[str] = Field(
        default_factory=list,
        description="Feature names or wildcard patterns to exclude from root cause analysis.",
    )
    asset_name: Optional[str] = Field(
        None, description="Optional asset identifier to use when persisting prediction artefacts."
    )
    debug_plots: Optional[bool] = Field(
        None, description="Override the configuration setting that controls debug plot generation."
    )

    @validator("ignore_features", pre=True)
    def _coerce_ignore_features(cls, value: Any) -> List[str]:  # pragma: no cover - validation logic
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class NarrativeRequest(PredictionRequest):
    """Request body for the LLM powered narrative endpoint."""

    llm: LLMConfiguration = Field(
        ..., description="Configuration describing which chat model should be used for generation."
    )
    enable_web_search: bool = Field(
        False,
        description="When True the service queries Perplexity for each event to enrich the narrative.",
    )
    perplexity_model: Optional[str] = Field(
        "sonar-small-chat",
        description="Model identifier to use when calling the Perplexity API.",
    )
    web_search_query_template: str = Field(
        "Wind turbine fault during {start} to {end} involving sensors {sensors}",
        description="Template used to craft web search prompts for each event.",
    )


class PredictionSummary(BaseModel):
    total_samples: int
    anomaly_samples: int
    critical_samples: int
    event_count: int


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    data_path: str
    asset_name: str
    ignore_features: List[str]
    artifact_directory: Optional[str]
    summary: PredictionSummary
    events: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"


class NarrativeEvent(BaseModel):
    event_id: int
    start: Optional[str]
    end: Optional[str]
    duration_minutes: float
    severity: str
    severity_reason: str
    sensors: List[str]
    sensor_scores: Dict[str, float]
    potential_root_cause: Optional[str]
    narrative: str


class NarrativeResponse(BaseModel):
    model_name: str
    model_version: str
    asset_name: str
    data_path: str
    summary: PredictionSummary
    configuration_summary: str
    global_summary: str
    events: List[NarrativeEvent]
    required_api_keys: List[str]


settings = get_settings()
model_registry = ModelRegistry(
    root_directory=settings.model_store.root_directory,
    default_version_strategy=settings.model_store.default_version_strategy,
)

_UPLOAD_ROOT = settings.prediction.output_directory
if _UPLOAD_ROOT is None:
    _UPLOAD_ROOT = Path.cwd() / "artifacts"
_UPLOAD_ROOT = Path(_UPLOAD_ROOT).resolve()
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_DATASET_UPLOAD_DIR = _UPLOAD_ROOT / "uploads"
_DATASET_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_filename(filename: Optional[str]) -> str:
    """Return a filesystem-safe filename derived from ``filename``."""

    if not filename:
        return "dataset.csv"
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


async def _persist_upload(file: UploadFile, prediction_id: str) -> Path:
    """Persist the uploaded file to disk and return the saved path."""

    target_dir = _DATASET_UPLOAD_DIR / prediction_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _normalise_filename(file.filename)
    try:
        with target_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    finally:
        await file.close()
    return target_path


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load the uploaded dataset into a DataFrame based on its suffix."""

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - optional dependency not installed
        raise HTTPException(
            status_code=400,
            detail="Support for the uploaded file format requires optional dependencies to be installed.",
        ) from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode the uploaded file: {exc}") from exc

    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, JSON, or Parquet data.")


def _serialise_timestamp(value: Any) -> Optional[str]:
    """Convert timestamp-like values into ISO-8601 strings when possible."""

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().astimezone(timezone.utc).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return str(value)


def _infer_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of a timestamp column in ``df``."""

    for column in df.columns:
        lowered = column.lower()
        if "time" in lowered or "timestamp" in lowered:
            return column
    return None


def _build_anomaly_events(df: pd.DataFrame, timestamp_column: Optional[str]) -> List[Dict[str, Any]]:
    """Derive simple anomaly-style events from numeric columns in ``df``."""

    numeric = df.select_dtypes(include=["number"])  # type: ignore[arg-type]
    if numeric.empty:
        return []

    timestamps = None
    if timestamp_column and timestamp_column in df.columns:
        timestamps = df[timestamp_column]

    events: List[Dict[str, Any]] = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue

        max_index = int(series.idxmax())
        peak_value = float(series.iloc[max_index])
        mean_value = float(series.mean())
        std_value = float(series.std(ddof=0))
        z_score = (peak_value - mean_value) / std_value if std_value else 0.0

        severity = "low"
        if z_score >= 2.5:
            severity = "high"
        elif z_score >= 1.5:
            severity = "medium"

        timestamp = None
        if timestamps is not None and max_index < len(timestamps):
            timestamp = _serialise_timestamp(timestamps.iloc[max_index])

        events.append(
            {
                "timestamp": timestamp,
                "metric": column,
                "score": round(peak_value, 6),
                "severity": severity,
                "message": f"Peak value detected for {column}",
            }
        )

    events.sort(key=lambda item: item["score"], reverse=True)
    return events


def _build_root_cause(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate lightweight root-cause style insights from ``events``."""

    if not events:
        return []

    total_score = sum(event["score"] for event in events) or 1.0
    root_cause: List[Dict[str, Any]] = []
    for event in events[:5]:
        contribution = event["score"] / total_score
        root_cause.append(
            {
                "metric": event["metric"],
                "importance": round(contribution, 4),
                "message": f"{event['metric']} accounts for {contribution:.0%} of the observed deviation.",
            }
        )
    return root_cause


def _build_narrative(
    record: PredictionRecord, df: pd.DataFrame, events: List[Dict[str, Any]], timestamp_column: Optional[str]
) -> str:
    """Craft a concise narrative describing the processed dataset."""

    row_count = len(df.index)
    sensor_columns = df.select_dtypes(include=["number"]).columns
    sensor_count = len(sensor_columns)
    high_severity = sum(1 for event in events if event["severity"] == "high")
    medium_severity = sum(1 for event in events if event["severity"] == "medium")

    time_hint = "timestamp column" if timestamp_column else "no explicit timestamp"

    return (
        f"Batch '{record.batch_name}' was processed with {row_count} samples across {sensor_count} numeric sensors "
        f"and {time_hint}. The analysis flagged {len(events)} potential anomalies "
        f"({high_severity} high, {medium_severity} medium severity) for follow-up."
    )


def _populate_prediction(record: PredictionRecord, df: pd.DataFrame) -> None:
    """Populate ``record`` with derived analytics from ``df``."""

    timestamp_column = _infer_timestamp_column(df)
    events = _build_anomaly_events(df, timestamp_column)
    record.anomalies = events
    record.critical_events = [event for event in events if event["severity"] == "high"][:5]
    record.root_causes = _build_root_cause(events)
    record.narrative = _build_narrative(record, df, events, timestamp_column)
    record.status = "ready"


def _get_prediction(prediction_id: str) -> PredictionRecord:
    """Return a stored prediction record or raise a 404 error."""

    record = PREDICTION_STORE.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return record


def _event_source_response(events: Iterable[Dict[str, Any]], error: Optional[str] = None) -> EventSourceResponse:
    """Create an :class:`EventSourceResponse` streaming ``events`` to the client."""

    async def event_generator() -> AsyncGenerator[str, None]:  # pragma: no cover - exercised via integration tests
        if error:
            payload = json.dumps({"status": "error", "detail": error})
            yield f"event: error\ndata: {payload}\n\n"
            return

        sent = False
        for item in events:
            sent = True
            yield f"data: {json.dumps(item)}\n\n"
        if not sent:
            yield f"data: {json.dumps({'status': 'ready'})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'ready'})}\n\n"

    return EventSourceResponse(event_generator())


def _json_or_sse_response(
    request: Request, events: List[Dict[str, Any]], *, error: Optional[str] = None
) -> Response:
    """Return either JSON or SSE data depending on the request Accept header."""

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return _event_source_response(events, error)
    if error:
        raise HTTPException(status_code=500, detail=error)
    return JSONResponse(events)

app = FastAPI(title="Energy Fault Detector", version="1.0.0")


class DatasetNarrativeRequest(BaseModel):
    """Payload for the lightweight narrative endpoint used by the UI."""

    prediction_id: str = Field(..., description="Identifier returned by the dataset upload endpoint.")


@app.post("/api/predictions")
async def upload_prediction(
    file: UploadFile = File(...),
    batch_name: str = Form(...),
    notes: str = Form(""),
) -> Dict[str, str]:
    """Persist an uploaded dataset and derive quick insights for the UI."""

    prediction_id = uuid4().hex
    logger.info("Received dataset upload for batch '%s' (prediction %s)", batch_name, prediction_id)

    try:
        saved_path = await _persist_upload(file, prediction_id)
        dataframe = _load_dataset(saved_path)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to persist uploaded dataset %s", prediction_id)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded dataset.") from exc

    record = PredictionRecord(
        prediction_id=prediction_id,
        batch_name=batch_name,
        notes=notes or None,
        file_path=saved_path,
    )

    try:
        _populate_prediction(record, dataframe)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Dataset processing failed for prediction %s", prediction_id)
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to analyse dataset: {exc}") from exc

    PREDICTION_STORE[prediction_id] = record
    logger.info(
        "Stored prediction %s with %s anomaly candidates (%s critical).",
        prediction_id,
        len(record.anomalies),
        len(record.critical_events),
    )

    return {"prediction_id": prediction_id}


@app.get("/webhooks/anomalies")
async def stream_anomalies(request: Request, prediction_id: str) -> Response:
    """Stream or return anomaly events for a given ``prediction_id``."""

    record = _get_prediction(prediction_id)
    return _json_or_sse_response(request, record.anomalies, error=record.error)


@app.get("/webhooks/critical")
async def stream_critical_events(request: Request, prediction_id: str) -> Response:
    """Provide critical event data for ``prediction_id``."""

    record = _get_prediction(prediction_id)
    return _json_or_sse_response(request, record.critical_events, error=record.error)


@app.get("/webhooks/rca")
async def stream_root_cause(request: Request, prediction_id: str) -> Response:
    """Provide root-cause analysis updates for ``prediction_id``."""

    record = _get_prediction(prediction_id)
    return _json_or_sse_response(request, record.root_causes, error=record.error)


@app.post("/api/narratives")
def generate_dataset_narrative(request: DatasetNarrativeRequest) -> Dict[str, str]:
    """Return the cached narrative summary for ``prediction_id``."""

    record = _get_prediction(request.prediction_id)
    if record.status != "ready":
        raise HTTPException(status_code=409, detail="Prediction is still processing. Please retry shortly.")

    summary = record.narrative or "Narrative not available for this dataset."
    return {"summary": summary}


@app.get("/api/predictions/{prediction_id}/report")
def download_prediction_report(prediction_id: str) -> Response:
    """Return a simple CSV report derived from the stored prediction insights."""

    record = _get_prediction(prediction_id)
    if record.status != "ready":
        raise HTTPException(status_code=409, detail="Prediction is still processing.")

    buffer = io.StringIO()
    fieldnames = ["metric", "score", "severity", "message", "timestamp"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for event in record.anomalies:
        writer.writerow(
            {
                "metric": event.get("metric"),
                "score": event.get("score"),
                "severity": event.get("severity"),
                "message": event.get("message"),
                "timestamp": event.get("timestamp", ""),
            }
        )

    csv_content = buffer.getvalue()
    buffer.close()

    headers = {"Content-Disposition": f"attachment; filename=prediction-{prediction_id}.csv"}
    return Response(content=csv_content, media_type="text/csv", headers=headers)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Simple endpoint to verify that the service is running."""

    return HealthResponse()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Execute the quick fault detector in prediction mode."""

    result_payload = _execute_prediction_pipeline(request)
    prediction_results = result_payload["prediction_results"]
    event_metadata = result_payload["event_metadata"]
    resolved_version = result_payload["resolved_version"]
    asset_name = result_payload["asset_name"]
    ignore_patterns = result_payload["ignore_patterns"]
    artifact_directory = result_payload["artifact_directory"]
    data_path = result_payload["data_path"]
    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    total_samples = int(len(predicted_anomalies))
    anomaly_samples = int(predicted_anomalies.get("anomaly", pd.Series(dtype=bool)).fillna(False).sum())
    critical_samples = int(
        predicted_anomalies.get("critical_event", pd.Series(dtype=bool)).fillna(False).sum()
    )
    event_count = int(len(event_metadata)) if event_metadata is not None else 0

    response = PredictionResponse(
        model_name=request.model_name,
        model_version=resolved_version,
        data_path=str(data_path),
        asset_name=asset_name,
        ignore_features=ignore_patterns,
        artifact_directory=str(artifact_directory) if artifact_directory is not None else None,
        summary=PredictionSummary(
            total_samples=total_samples,
            anomaly_samples=anomaly_samples,
            critical_samples=critical_samples,
            event_count=event_count,
        ),
        events=_dataframe_to_records(event_metadata),
    )

    return response


@app.post("/narrative", response_model=NarrativeResponse)
async def generate_narrative(request: NarrativeRequest) -> NarrativeResponse:
    """Run the prediction pipeline and generate an LLM powered narrative."""

    data_path = Path(request.data_path).expanduser()
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Data file '{data_path}' does not exist.")

    result_payload = await run_in_threadpool(_execute_prediction_pipeline, request)

    prediction_results = result_payload["prediction_results"]
    event_metadata = result_payload["event_metadata"]
    event_analysis = result_payload["event_analysis"]
    resolved_version = result_payload["resolved_version"]
    asset_name = result_payload["asset_name"]

    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    total_samples = int(len(predicted_anomalies))
    anomaly_samples = int(predicted_anomalies.get("anomaly", pd.Series(dtype=bool)).fillna(False).sum())
    critical_samples = int(
        predicted_anomalies.get("critical_event", pd.Series(dtype=bool)).fillna(False).sum()
    )
    event_count = int(len(event_metadata)) if event_metadata is not None else 0

    base_insights = derive_event_insights(event_metadata, event_analysis, predicted_anomalies)

    root_cause_results: List[Optional[str]]
    search_client = None
    if request.enable_web_search:
        search_client = PerplexityClient(model=request.perplexity_model)
        if not search_client.is_configured():
            logger.warning(
                "Perplexity API key missing while web search requested; proceeding without enrichment."
            )
            root_cause_results = [None] * len(base_insights)
        else:
            queries: List[str] = []
            for insight in base_insights:
                serialisable = insight.serialisable()
                sensors = serialisable.get("sensors") or []
                sensors_text = ", ".join(sensors) if sensors else "unknown sensors"
                query = request.web_search_query_template.format(
                    start=serialisable.get("start"),
                    end=serialisable.get("end"),
                    sensors=sensors_text,
                    severity=insight.severity,
                )
                queries.append(query)
            root_cause_results = await gather_root_cause_hypotheses(
                queries, search_client.search if search_client else None
            )
    else:
        root_cause_results = [None] * len(base_insights)

    event_insights = derive_event_insights(
        event_metadata,
        event_analysis,
        predicted_anomalies,
        root_cause_hypotheses=root_cause_results,
    )

    critical_events = sum(1 for insight in event_insights if insight.severity == "critical")
    glitch_events = sum(1 for insight in event_insights if insight.severity == "temporary_glitch")

    summary = PredictionSummary(
        total_samples=total_samples,
        anomaly_samples=anomaly_samples,
        critical_samples=critical_samples,
        event_count=event_count,
    )

    llm = create_chat_model(request.llm)
    pipeline = NarrativePipeline(llm)
    configuration_context = summarise_configuration(prediction_results)
    metadata = {
        "asset_name": asset_name,
        "model_name": request.model_name,
        "model_version": resolved_version,
    }

    narrative_context = NarrativeContext(
        total_anomalies=anomaly_samples,
        critical_events=critical_events,
        glitch_events=glitch_events,
        event_insights=event_insights,
        configuration_context=configuration_context,
        metadata=metadata,
    )

    narrative_result = await pipeline.arun(narrative_context)

    events: List[NarrativeEvent] = []
    for insight, story in zip(event_insights, narrative_result.event_narratives):
        serialisable = insight.serialisable()
        events.append(
            NarrativeEvent(
                event_id=insight.event_id,
                start=serialisable.get("start"),
                end=serialisable.get("end"),
                duration_minutes=insight.duration_minutes,
                severity=insight.severity,
                severity_reason=insight.severity_reason,
                sensors=insight.sensors,
                sensor_scores=insight.sensor_scores,
                potential_root_cause=insight.potential_root_cause,
                narrative=story,
            )
        )

    required_api_keys: List[str] = []
    provider = request.llm.provider
    if provider == "openai":
        required_api_keys.append("OPENAI_API_KEY or llm.api_key")
    elif provider == "bedrock":
        required_api_keys.append("AWS credentials with Bedrock permissions")
    if request.enable_web_search:
        required_api_keys.append("PERPLEXITY_API_KEY")

    return NarrativeResponse(
        model_name=request.model_name,
        model_version=resolved_version,
        asset_name=asset_name,
        data_path=str(result_payload["data_path"]),
        summary=summary,
        configuration_summary=narrative_result.configuration_summary,
        global_summary=narrative_result.global_summary,
        events=events,
        required_api_keys=required_api_keys,
    )


__all__ = [
    "app",
    "PredictionRequest",
    "NarrativeRequest",
    "PredictionResponse",
    "NarrativeResponse",
    "PredictionSummary",
]
