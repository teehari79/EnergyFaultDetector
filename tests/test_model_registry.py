"""Tests for the model registry helper."""

from pathlib import Path

import pytest

from energy_fault_detector.api.model_registry import ModelNotFoundError, ModelRegistry


def _create_model_dir(base: Path, model_name: str, version: str) -> Path:
    path = base / model_name / version
    (path / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("model: test", encoding="utf-8")
    return path


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
