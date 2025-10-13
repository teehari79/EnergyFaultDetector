"""FastAPI application exposing prediction endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from energy_fault_detector.quick_fault_detection import quick_fault_detector

from .model_registry import ModelNotFoundError, ModelRegistry
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

    try:
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
        prediction_results, event_metadata, _ = quick_fault_detector(
            csv_data_path=None,
            csv_test_data_path=str(data_path),
            train_test_column_name=settings.prediction.train_test_column,
            train_test_mapping=settings.prediction.train_test_mapping or None,
            time_column_name=settings.prediction.time_column,
            status_data_column_name=settings.prediction.status_column,
            status_mapping=settings.prediction.status_mapping or None,
            min_anomaly_length=settings.prediction.min_anomaly_length,
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

    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    total_samples = int(len(predicted_anomalies))
    anomaly_samples = int(predicted_anomalies.get("anomaly", pd.Series(dtype=bool)).fillna(False).sum())
    critical_samples = int(
        predicted_anomalies.get("critical_event", pd.Series(dtype=bool)).fillna(False).sum()
    )
    event_count = int(len(event_metadata)) if event_metadata is not None else 0

    artifact_directory = _derive_artifact_directory(save_root, data_path, asset_name)

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


__all__ = ["app", "PredictionRequest", "PredictionResponse", "PredictionSummary"]
