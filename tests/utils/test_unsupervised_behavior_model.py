"""Tests for the unsupervised behaviour model utility."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from energy_fault_detector.utils.unsupervised_behavior_model import (
    SensorRange,
    UnsupervisedBehaviorModel,
)


def _create_synthetic_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {
            "power": rng.normal(loc=1.0, scale=0.05, size=512),
            "wind_speed": rng.normal(loc=12.0, scale=0.5, size=512),
        }
    )
    return base


def test_sensor_range_contains_works_for_bounds() -> None:
    sensor_range = SensorRange(lower=0.0, upper=10.0)
    series = pd.Series([-1.0, 0.0, 5.0, 10.0, 11.0])

    result = sensor_range.contains(series)

    np.testing.assert_array_equal(result.values, np.array([False, True, True, True, False]))


def test_fit_computes_normal_ranges() -> None:
    data = _create_synthetic_data()

    model = UnsupervisedBehaviorModel(contamination=0.05, quantile_bounds=(0.05, 0.95), random_state=1)
    model.fit(data)
    normal_ranges = model.get_normal_ranges()

    assert list(normal_ranges.index) == ["power", "wind_speed"]
    # Verify that medians are close to the expected centre point.
    assert abs(normal_ranges.loc["power", "median"] - 1.0) < 0.01
    assert abs(normal_ranges.loc["wind_speed", "median"] - 12.0) < 0.1


def test_predict_flags_outliers() -> None:
    data = _create_synthetic_data()
    model = UnsupervisedBehaviorModel(contamination=0.05, quantile_bounds=(0.05, 0.95), random_state=1)
    model.fit(data)

    evaluation = pd.DataFrame(
        {
            "power": [1.02, 3.0],
            "wind_speed": [12.1, 5.0],
        }
    )

    predictions = model.predict(evaluation)

    assert predictions.loc[0, "is_anomaly"] is False
    assert predictions.loc[1, "is_anomaly"] is True
    assert predictions.loc[1, "range_violation_count"] >= 1
    assert "wind_speed" in predictions.loc[1, "violated_sensors"]

