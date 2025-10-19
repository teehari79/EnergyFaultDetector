"""Tests for the model registry helper."""

from pathlib import Path

import pytest

from energy_fault_detector.api.model_registry import ModelNotFoundError, ModelRegistry


def _create_model_dir(base: Path, model_name: str, version: str) -> Path:
    path = base / model_name / version
    (path / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model: test", encoding="utf-8")
    return path


def _create_asset_model(base: Path, asset_number: str, version: str) -> Path:
    model_dir = base / f"asset_{asset_number}" / "models" / version
    (model_dir / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text("model: test", encoding="utf-8")
    for name in ("data_preprocessor", "autoencoder", "threshold_selector", "anomaly_score"):
        (model_dir / name).mkdir(parents=True, exist_ok=True)
    return model_dir


def test_resolve_latest_version(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)
    _create_model_dir(tmp_path, "turbine", "20230101")
    latest = _create_model_dir(tmp_path, "turbine", "20240202")

    resolved_path, version = registry.resolve("turbine")

    assert resolved_path == latest
    assert version == "20240202"


def test_resolve_specific_version(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)
    expected = _create_model_dir(tmp_path, "asset", "v1")

    resolved_path, version = registry.resolve("asset", "v1")

    assert resolved_path == expected
    assert version == "v1"


def test_resolve_versionless_model(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)
    path = tmp_path / "single"
    (path / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model: test", encoding="utf-8")

    resolved_path, version = registry.resolve("single")

    assert resolved_path == path
    assert version == "single"


def test_resolve_missing_model(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.resolve("unknown")


def test_resolve_from_asset_name_uses_latest_version(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)
    _create_asset_model(tmp_path, "34", "20240101_000000")
    latest = _create_asset_model(tmp_path, "34", "20240202_120000")

    resolved_path, resolved_version = registry.resolve_from_asset_name("farm-c-asset-34")

    assert resolved_path == latest
    assert resolved_version == "20240202_120000"


def test_resolve_from_asset_name_supports_explicit_version(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)
    expected = _create_asset_model(tmp_path, "12", "v1")
    _create_asset_model(tmp_path, "12", "v2")

    resolved_path, resolved_version = registry.resolve_from_asset_name(
        "farm-c-asset-12", model_version="v1"
    )

    assert resolved_path == expected
    assert resolved_version == "v1"


def test_resolve_from_asset_name_requires_numeric_identifier(tmp_path):
    registry = ModelRegistry(root_directory=tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.resolve_from_asset_name("farm-c-asset")
