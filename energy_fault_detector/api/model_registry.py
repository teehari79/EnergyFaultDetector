"""Utilities for locating trained model artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested model or version does not exist."""


ASSET_COMPONENT_NAMES = {"data_preprocessor", "autoencoder", "threshold_selector", "anomaly_score"}


def _looks_like_model_directory(path: Path) -> bool:
    """Heuristic to determine whether ``path`` contains model artefacts."""

    if not path.is_dir():
        return False

    if (path / "config.yaml").exists():
        return True

    child_directories = {child.name for child in path.iterdir() if child.is_dir()}
    return bool(ASSET_COMPONENT_NAMES & child_directories)


def _iter_asset_model_directories(asset_root: Path) -> List[Path]:
    """Return directories under ``asset_root`` that contain model artefacts."""

    stack = [asset_root]
    candidates: List[Path] = []

    while stack:
        current = stack.pop()
        if not current.is_dir():
            continue

        if _looks_like_model_directory(current) and current.name not in ASSET_COMPONENT_NAMES:
            candidates.append(current)
            continue

        for child in current.iterdir():
            if child.is_dir():
                stack.append(child)

    return candidates


_ASSET_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _extract_asset_number(asset_name: str) -> Optional[str]:
    """Extract the trailing numeric identifier from ``asset_name`` if present."""

    match = _ASSET_NUMBER_PATTERN.search(asset_name)
    if match is None:
        return None
    return match.group(1)


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

    def resolve_asset(self, asset_number: str, model_version: Optional[str] = None) -> Tuple[Path, str]:
        """Resolve a model path for ``asset_number`` stored under the registry root."""

        asset_root = self.root_directory / f"asset_{asset_number}"
        if not asset_root.exists():
            raise ModelNotFoundError(
                f"Asset '{asset_number}' was not found under '{self.root_directory}'."
            )

        candidates = _iter_asset_model_directories(asset_root)
        if not candidates:
            raise ModelNotFoundError(
                f"No model artefacts discovered for asset '{asset_number}' in '{asset_root}'."
            )

        if model_version:
            for candidate in candidates:
                if candidate.name == model_version:
                    return candidate, candidate.name
            raise ModelNotFoundError(
                f"Asset '{asset_number}' does not provide a model version named '{model_version}'."
            )

        selected = _sort_versions(candidates)[-1]
        return selected, selected.name

    def resolve_from_asset_name(
        self, asset_name: str, model_version: Optional[str] = None
    ) -> Tuple[Path, str]:
        """Resolve a model path using the numeric identifier embedded in ``asset_name``."""

        asset_number = _extract_asset_number(asset_name)
        if asset_number is None:
            raise ModelNotFoundError(
                f"Asset name '{asset_name}' does not contain a numeric identifier."
            )

        return self.resolve_asset(asset_number, model_version)

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
