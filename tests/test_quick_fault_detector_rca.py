"""Ensure that additional RCA ignore patterns are respected during prediction."""

from __future__ import annotations

from typing import List

import pytest

pd = pytest.importorskip("pandas")

from energy_fault_detector.quick_fault_detection.quick_fault_detector import quick_fault_detector


class DummyResult:
    def __init__(self) -> None:
        self.predicted_anomalies = pd.DataFrame({"anomaly": [False], "critical_event": [False]})

    def save(self, *_args, **_kwargs) -> None:
        return None


class DummyFaultDetector:
    instances: List["DummyFaultDetector"] = []

    def __init__(self, config, **_kwargs) -> None:  # noqa: D401 - matches interface
        self.config = config
        self.instances.append(self)

    def load_models(self, *_args, **_kwargs) -> None:
        return None

    def predict(self, *_args, **_kwargs):
        return DummyResult()


def test_quick_fault_detector_extends_ignore_features(monkeypatch, tmp_path):
    DummyFaultDetector.instances = []

    def fake_load_train_test_data(**_kwargs):
        data = pd.DataFrame({"a": [1.0, 2.0]})
        return data, pd.Series([True, False]), data

    def fake_create_events(**_kwargs):
        return pd.DataFrame(), []

    def fake_analyze_event(**_kwargs):
        return pd.Series(dtype=float), pd.DataFrame()

    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector.load_train_test_data",
        fake_load_train_test_data,
    )
    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector.create_events",
        fake_create_events,
    )
    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector.analyze_event",
        fake_analyze_event,
    )
    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector.generate_output_plots",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector._save_prediction_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "energy_fault_detector.quick_fault_detection.quick_fault_detector.FaultDetector",
        DummyFaultDetector,
    )

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n2\n", encoding="utf-8")

    quick_fault_detector(
        csv_data_path=None,
        csv_test_data_path=str(csv_path),
        mode="predict",
        model_path=str(tmp_path),
        rca_ignore_features=["feature_a"],
    )

    assert DummyFaultDetector.instances, "FaultDetector was not instantiated"
    ignore_features = DummyFaultDetector.instances[0].config.config_dict["root_cause_analysis"]["ignore_features"]
    assert "feature_a" in ignore_features
