"""Unsupervised behaviour modelling for wind turbine sensor data.

This module provides a lightweight, fully unsupervised workflow that can be
used when only raw sensor logs of a single wind turbine are available. The
``UnsupervisedBehaviorModel`` learns robust per-sensor operating ranges and a
multivariate anomaly detector (Isolation Forest) that jointly flag
observations deviating from historic patterns. It can be reused for any wind
farm by passing a :class:`pandas.DataFrame` with numerical sensor columns.

Example
-------
>>> import pandas as pd
>>> from energy_fault_detector.utils.unsupervised_behavior_model import (
...     UnsupervisedBehaviorModel,
... )
>>> # sensor logs of a single turbine (no labels required)
>>> df = pd.DataFrame({"power": [1.1, 1.2, 5.5], "wind_speed": [7.5, 7.4, 25]})
>>> model = UnsupervisedBehaviorModel().fit(df)
>>> model.get_normal_ranges()
          lower  median  upper
power       ...     ...    ...
wind_speed  ...     ...    ...
>>> predictions = model.predict(df)
>>> predictions["is_anomaly"].tolist()
[False, False, True]

The implementation purposely favours transparency over complex modelling so
that domain experts can inspect the inferred ranges and understand why a
record has been labelled anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class SensorRange:
    """Container for the inferred normal operating range of a sensor."""

    lower: float
    upper: float

    def contains(self, values: pd.Series) -> pd.Series:
        """Return a boolean series indicating which values fall inside the range."""

        return (values >= self.lower) & (values <= self.upper)


class UnsupervisedBehaviorModel:
    """Identify anomalous sensor behaviour without labelled data.

    The model combines per-sensor robust statistics with an Isolation Forest to
    flag outliers. Per-sensor ranges are derived from empirical quantiles and an
    inter-quantile margin so the approach remains interpretable. The Isolation
    Forest captures multivariate anomalies that may not breach individual
    thresholds but still indicate abnormal behaviour when considering the
    sensors jointly.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies in the historic data. This is forwarded
        to the Isolation Forest.
    quantile_bounds:
        Low/high quantiles that represent typical behaviour. A Tukey-style
        whisker is added on top of this interval to soften the range.
    random_state:
        Optional seed for deterministic Isolation Forest behaviour.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        quantile_bounds: Tuple[float, float] = (0.01, 0.99),
        random_state: Optional[int] = 42,
    ) -> None:
        if not 0 < contamination < 0.5:
            raise ValueError("`contamination` must be between 0 and 0.5.")

        low_q, high_q = quantile_bounds
        if not 0.0 < low_q < high_q < 1.0:
            raise ValueError(
                "`quantile_bounds` must satisfy 0 < low_quantile < high_quantile < 1."
            )

        self.contamination: float = contamination
        self.quantile_bounds: Tuple[float, float] = (low_q, high_q)
        self.random_state: Optional[int] = random_state

        self._scaler: Optional[RobustScaler] = None
        self._model: Optional[IsolationForest] = None
        self._feature_names: Optional[pd.Index] = None
        self._normal_ranges: Dict[str, SensorRange] = {}
        self._feature_medians: Optional[pd.Series] = None

    @staticmethod
    def _select_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric columns, coercing boolean values to integers."""

        numeric = data.select_dtypes(include=[np.number, "bool"]).copy()
        for column in numeric.columns:
            if numeric[column].dtype == bool:
                numeric[column] = numeric[column].astype(float)
        return numeric.apply(pd.to_numeric, errors="coerce")

    def fit(self, data: pd.DataFrame) -> "UnsupervisedBehaviorModel":
        """Learn per-sensor ranges and fit the Isolation Forest.

        Parameters
        ----------
        data:
            A DataFrame containing historic sensor readings for a single wind
            turbine. Only numeric columns are used. Missing values are imputed
            with the column medians.
        """

        numeric_data = self._select_numeric_features(data)
        if numeric_data.empty:
            raise ValueError("`data` must contain at least one numeric column.")

        self._feature_names = numeric_data.columns
        self._feature_medians = numeric_data.median()

        self._normal_ranges = {}
        low_q, high_q = self.quantile_bounds
        for column in self._feature_names:
            series = numeric_data[column].dropna()
            if series.empty:
                # Degenerate column, fall back to the median and allow a narrow band.
                median_value = self._feature_medians[column]
                margin = max(abs(median_value) * 0.05, 1e-9)
                self._normal_ranges[column] = SensorRange(
                    lower=median_value - margin,
                    upper=median_value + margin,
                )
                continue

            low = float(series.quantile(low_q))
            high = float(series.quantile(high_q))
            iqr = high - low
            if np.isclose(iqr, 0.0):
                std = float(series.std(ddof=0))
                margin = 3.0 * std if not np.isclose(std, 0.0) else max(
                    abs(series.iloc[0]) * 0.05,
                    1e-9,
                )
            else:
                margin = 1.5 * iqr

            self._normal_ranges[column] = SensorRange(lower=low - margin, upper=high + margin)

        filled_data = numeric_data.fillna(self._feature_medians)
        self._scaler = RobustScaler().fit(filled_data)
        scaled = self._scaler.transform(filled_data)

        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(scaled)

        return self

    @property
    def feature_names(self) -> pd.Index:
        if self._feature_names is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self._feature_names

    def get_normal_ranges(self) -> pd.DataFrame:
        """Return the learnt normal operating ranges for each sensor."""

        if self._feature_medians is None:
            raise RuntimeError("The model has not been fitted yet.")

        records = []
        for name in self.feature_names:
            sensor_range = self._normal_ranges[name]
            records.append(
                {
                    "sensor": name,
                    "lower": sensor_range.lower,
                    "median": self._feature_medians[name],
                    "upper": sensor_range.upper,
                }
            )
        return pd.DataFrame.from_records(records).set_index("sensor")

    def _prepare_prediction_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._feature_medians is None:
            raise RuntimeError("The model has not been fitted yet.")

        numeric_data = self._select_numeric_features(data)
        missing_columns = [col for col in self.feature_names if col not in numeric_data]
        for column in missing_columns:
            numeric_data[column] = self._feature_medians[column]
        numeric_data = numeric_data[self.feature_names]
        return numeric_data.fillna(self._feature_medians)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Label observations as normal or anomalous.

        Parameters
        ----------
        data:
            DataFrame with the same sensor columns that were used during
            :meth:`fit`. Additional columns are ignored. The returned DataFrame
            contains a boolean ``is_anomaly`` column together with diagnostic
            information.
        """

        if self._model is None or self._scaler is None:
            raise RuntimeError("The model has not been fitted yet.")

        numeric_data = self._prepare_prediction_frame(data)
        scaled = self._scaler.transform(numeric_data)

        anomaly_scores = -self._model.score_samples(scaled)
        isolation_flags = self._model.predict(scaled) == -1

        # Range violations per sensor.
        range_flags = {}
        for column in self.feature_names:
            sensor_range = self._normal_ranges[column]
            range_flags[column] = ~sensor_range.contains(numeric_data[column])
        range_violation_df = pd.DataFrame(range_flags)
        violation_counts = range_violation_df.sum(axis=1)

        def violated_columns(row: pd.Series) -> List[str]:
            return row.index[row].tolist()

        violations = range_violation_df.apply(violated_columns, axis=1)

        result = pd.DataFrame(index=data.index)
        result["anomaly_score"] = anomaly_scores
        result["isolation_forest_flag"] = isolation_flags
        result["range_violation_count"] = violation_counts
        result["violated_sensors"] = violations
        result["is_anomaly"] = isolation_flags | (violation_counts > 0)

        return result

    def explain(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return prediction results joined with the original sensor values.

        This helper simplifies reporting by concatenating ``predict`` output and
        the respective sensor readings for quick inspection.
        """

        prediction = self.predict(data)
        numeric_data = self._prepare_prediction_frame(data)
        return pd.concat([numeric_data, prediction], axis=1)


def aggregate_wind_farm_data(
    dataset_root: str | PathLike[str],
    wind_farm: str = "B",
    statistics: Optional[Sequence[str]] = None,
    index_column: str = "id",
) -> pd.DataFrame:
    """Aggregate Care-to-Compare data for a specific wind farm.

    The Care-to-Compare (C2C) dataset is organised per anomaly event. For
    unsupervised modelling it is convenient to concatenate all prediction
    segments into a single DataFrame. The function keeps only sensor columns and
    drops the metadata columns. Normal/anomalous status labels are intentionally
    ignored to emulate a purely unsupervised scenario.

    Parameters
    ----------
    dataset_root:
        Root path that contains the extracted C2C dataset.
    wind_farm:
        Select which wind farm ("A", "B" or "C") to load.
    statistics:
        Optional statistics suffixes passed to :class:`Care2CompareDataset` to
        control which sensor aggregates are loaded.
    index_column:
        Column used as index when reading the CSV files ("id" or "time_stamp").
    """

    from energy_fault_detector.evaluation.care2compare import Care2CompareDataset

    dataset = Care2CompareDataset(dataset_root)
    frames: List[pd.DataFrame] = []
    if statistics is None:
        stat_list: Optional[List[str]] = None
    elif isinstance(statistics, str):
        stat_list = [statistics]
    else:
        stat_list = list(statistics)

    for (event_data, _event_id) in dataset.iter_datasets(
        wind_farm=wind_farm, test_only=True, statistics=stat_list,
        index_column=index_column,
    ):
        sensor_data = event_data.drop(
            columns=[
                column
                for column in ["asset_id", "id", "time_stamp", "status_type_id", "train_test"]
                if column in event_data.columns
            ],
            errors="ignore",
        )
        frames.append(sensor_data)

    if not frames:
        raise ValueError(
            "No datasets were found. Verify that `dataset_root` points to the extracted Care-to-Compare data."
        )

    return pd.concat(frames, axis=0)


__all__ = [
    "SensorRange",
    "UnsupervisedBehaviorModel",
    "aggregate_wind_farm_data",
]

