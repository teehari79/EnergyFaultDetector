"""REST API utilities for the Energy Fault Detector package."""

from __future__ import annotations

from importlib import import_module
from typing import Any, List

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        module = import_module("energy_fault_detector.api.app")
        return module.app
    raise AttributeError(name)


def __dir__() -> List[str]:  # pragma: no cover - small helper
    return sorted(__all__)
