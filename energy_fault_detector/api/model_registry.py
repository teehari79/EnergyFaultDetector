"""Utilities for locating trained model artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested model or version does not exist."""


def _looks_like_model_directory(path: Path) -> bool:
    """Heuristic to determine whether ``path`` contains model artefacts."""

    if not path.is_dir():
        return False

    if (path / "config.yaml").exists():
        return True

    expected_components = {"data_preprocessor", "autoencoder", "threshold_selector", "anomaly_score"}
    child_directories = {child.name for child in path.iterdir() if child.is_dir()}
    return bool(expected_components & child_directories)


def _sort_versions(versions: Iterable[Path]) -> List[Path]:
    return sorted(versions, key=lambda version: version.name)


@dataclass
class ModelRegistry:
    """Simple model registry resolving model names to artefact locations."""

    root_directory: Path
    default_version_strategy: str = "latest"

    def __post_init__(self) -> None:
        self.root_directory = Path(self.root_directory).resolve()

    def resolve(self, model_name: str, model_version: Optional[str] = None) -> Tuple[Path, str]:
        """Return the filesystem path for ``model_name`` and ``model_version``."""

        model_root = self.root_directory / model_name
        if not model_root.exists():
            raise ModelNotFoundError(f"Model '{model_name}' was not found under '{self.root_directory}'.")

        if model_version:
            candidate = model_root / model_version
            if not _looks_like_model_directory(candidate):
                raise ModelNotFoundError(
                    f"Model '{model_name}' does not provide a version named '{model_version}'."
                )
            return candidate, model_version

        versions = [child for child in model_root.iterdir() if _looks_like_model_directory(child)]

        if versions:
            if self.default_version_strategy != "latest":
                raise ValueError(
                    f"Unsupported version selection strategy '{self.default_version_strategy}'."
                )
            selected = _sort_versions(versions)[-1]
            return selected, selected.name

        if _looks_like_model_directory(model_root):
            return model_root, model_root.name

        raise ModelNotFoundError(
            f"No model versions found for '{model_name}' in '{model_root}'. Ensure the expected directory structure exists."
        )

    def list_versions(self, model_name: str) -> List[str]:
        """Return all discovered versions for ``model_name``."""

        model_root = self.root_directory / model_name
        if not model_root.exists():
            raise ModelNotFoundError(f"Model '{model_name}' was not found under '{self.root_directory}'.")

        versions = [child for child in model_root.iterdir() if _looks_like_model_directory(child)]
        if versions:
            return [path.name for path in _sort_versions(versions)]

        if _looks_like_model_directory(model_root):
            return [model_root.name]

        return []


__all__ = ["ModelRegistry", "ModelNotFoundError"]
