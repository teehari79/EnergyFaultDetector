"""Settings loader for the REST API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("service_config.yaml")
CONFIG_ENV_VAR = "EFD_SERVICE_CONFIG"


@dataclass
class ModelStoreSettings:
    """Settings describing where trained models are stored."""

    root_directory: Path
    default_version_strategy: str = "latest"


@dataclass
class PredictionSettings:
    """Settings that control how predictions are executed."""

    config_path: Path
    time_column: Optional[str] = None
    status_column: Optional[str] = None
    status_mapping: Dict[str, bool] = field(default_factory=dict)
    train_test_column: Optional[str] = None
    train_test_mapping: Dict[str, bool] = field(default_factory=dict)
    min_anomaly_length: int = 18
    critical_event_min_length: Optional[int] = None
    critical_event_min_duration: Optional[object] = None
    debug_plots: bool = False
    output_directory: Optional[Path] = None
    default_asset_name: str = "{model_name}"
    default_ignore_features: List[str] = field(default_factory=list)


@dataclass
class TenantSecuritySettings:
    """Authentication material for a single tenant/organisation."""

    seed_token: str
    users: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecuritySettings:
    """Settings that control API authentication and encryption."""

    token_ttl_seconds: int = 900
    tenants: Dict[str, TenantSecuritySettings] = field(default_factory=dict)


@dataclass
class CORSSettings:
    """Settings that describe how the API should handle cross-origin requests."""

    allow_origins: List[str] = field(default_factory=list)


@dataclass
class APISettings:
    """Container for all API settings."""

    model_store: ModelStoreSettings
    prediction: PredictionSettings
    security: SecuritySettings = field(default_factory=SecuritySettings)
    cors: CORSSettings = field(default_factory=CORSSettings)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[Path]:
    """Resolve ``value`` relative to ``base_dir`` when it is not absolute."""

    if value in (None, ""):
        return None

    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_yaml(config_path: Path) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("The service configuration must be a YAML mapping/dictionary.")
    return data


@lru_cache()
def get_settings() -> APISettings:
    """Return the API settings, loading them from disk on first access."""

    configured_path = os.environ.get(CONFIG_ENV_VAR)
    config_path = Path(configured_path) if configured_path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Service configuration file '{config_path}' could not be found."
        )

    config_path = config_path.resolve()
    raw_config = _load_yaml(config_path)
    base_dir = config_path.parent

    model_store_cfg = raw_config.get("model_store", {}) or {}
    prediction_cfg = raw_config.get("prediction", {}) or {}
    security_cfg = raw_config.get("security", {}) or {}
    cors_cfg = raw_config.get("cors", {}) or {}

    model_root = _resolve_path(base_dir, model_store_cfg.get("root_directory", "./models"))
    if model_root is None:
        raise ValueError("A model store root directory must be provided in the configuration.")

    version_strategy = model_store_cfg.get("default_version_strategy", "latest")

    prediction_config_path = _resolve_path(
        base_dir, prediction_cfg.get("config_path", "../base_config.yaml")
    )
    if prediction_config_path is None or not prediction_config_path.exists():
        raise FileNotFoundError(
            f"Prediction configuration file '{prediction_config_path}' could not be found."
        )

    output_directory = _resolve_path(base_dir, prediction_cfg.get("output_directory"))

    default_asset_name = prediction_cfg.get("default_asset_name", "{model_name}")
    default_ignore = prediction_cfg.get("default_ignore_features") or []

    critical_event_cfg = prediction_cfg.get("critical_event") or {}
    min_critical_length = critical_event_cfg.get("min_consecutive_samples")
    if min_critical_length is not None:
        min_critical_length = int(min_critical_length)
    min_critical_duration = critical_event_cfg.get("min_duration")

    token_ttl_seconds = int(security_cfg.get("token_ttl_seconds", 900))
    tenant_cfg = security_cfg.get("tenants") or {}
    tenants: Dict[str, TenantSecuritySettings] = {}
    for org_id, raw_tenant in tenant_cfg.items():
        if not isinstance(raw_tenant, dict):
            raise ValueError(
                "Each tenant definition in the security configuration must be a mapping/dictionary."
            )
        seed_token = raw_tenant.get("seed_token")
        if not seed_token:
            raise ValueError(
                f"Security configuration for tenant '{org_id}' must define a non-empty seed_token."
            )
        users = dict(raw_tenant.get("users") or {})
        tenants[str(org_id)] = TenantSecuritySettings(seed_token=str(seed_token), users=users)

    allow_origins = cors_cfg.get("allow_origins") or []
    if isinstance(allow_origins, str):
        allow_origins = [allow_origins]

    settings = APISettings(
        model_store=ModelStoreSettings(
            root_directory=model_root,
            default_version_strategy=str(version_strategy),
        ),
        prediction=PredictionSettings(
            config_path=prediction_config_path,
            time_column=prediction_cfg.get("time_column"),
            status_column=prediction_cfg.get("status_column"),
            status_mapping=dict(prediction_cfg.get("status_mapping") or {}),
            train_test_column=prediction_cfg.get("train_test_column"),
            train_test_mapping=dict(prediction_cfg.get("train_test_mapping") or {}),
            min_anomaly_length=int(prediction_cfg.get("min_anomaly_length", 18)),
            critical_event_min_length=min_critical_length,
            critical_event_min_duration=min_critical_duration,
            debug_plots=bool(prediction_cfg.get("debug_plots", False)),
            output_directory=output_directory,
            default_asset_name=str(default_asset_name),
            default_ignore_features=list(default_ignore),
        ),
        security=SecuritySettings(
            token_ttl_seconds=token_ttl_seconds,
            tenants=tenants,
        ),
        cors=CORSSettings(allow_origins=list(allow_origins)),
    )

    return settings


__all__ = [
    "APISettings",
    "ModelStoreSettings",
    "PredictionSettings",
    "SecuritySettings",
    "CORSSettings",
    "TenantSecuritySettings",
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "get_settings",
]
