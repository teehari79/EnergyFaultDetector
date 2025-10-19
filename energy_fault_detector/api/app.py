"""FastAPI application exposing prediction endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.concurrency import run_in_threadpool

import pandas as pd
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Energy Fault Detector", version="1.0.0")


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
