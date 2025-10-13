import unittest
from datetime import datetime, timedelta

import pandas as pd

from energy_fault_detector.utils import analysis
from energy_fault_detector.root_cause_analysis.arcana_utils import calculate_mean_arcana_importances


class TestAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_create_anomaly_events_returns_correct_values(self):
        """Test if create_anomaly_events returns a DataFrame with correct values"""
        sensor_data = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]},
                                   index=pd.date_range('2022-01-01', freq='D', periods=3))
        predicted_anomalies = pd.DataFrame({'anomaly': [True, False, True]},
                                           index=sensor_data.index)
        event_data, _ = analysis.create_events(sensor_data=sensor_data,
                                               boolean_information=predicted_anomalies['anomaly'],
                                               min_event_length=1)
        expected_values = {'start': [datetime(2022, 1, 1), datetime(2022, 1, 3)],
                           'end': [datetime(2022, 1, 1), datetime(2022, 1, 3)],
                           'duration': [timedelta(days=0), timedelta(days=0)]}
        expected_result = pd.DataFrame(expected_values, index=[0, 1])
        pd.testing.assert_frame_equal(event_data, expected_result)

    def test_create_events_uses_duration_threshold(self):
        timestamps = pd.date_range('2022-01-01', periods=6, freq='30min')
        sensor_data = pd.DataFrame({'a': range(6)}, index=timestamps)
        anomalies = pd.Series([False, True, True, True, False, False], index=timestamps)

        meta, events = analysis.create_events(
            sensor_data=sensor_data,
            boolean_information=anomalies,
            min_event_length=5,
            min_event_duration='90min',
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(meta.iloc[0]['start'], timestamps[1])
        self.assertEqual(meta.iloc[0]['end'], timestamps[3])

    def test_create_events_returns_empty_when_criteria_not_met(self):
        timestamps = pd.date_range('2022-01-01', periods=4, freq='H')
        sensor_data = pd.DataFrame({'a': range(4)}, index=timestamps)
        anomalies = pd.Series([False, True, True, False], index=timestamps)

        meta, events = analysis.create_events(
            sensor_data=sensor_data,
            boolean_information=anomalies,
            min_event_length=5,
            min_event_duration='3h',
        )

        self.assertTrue(meta.empty)
        self.assertEqual(events, [])

    def test_get_criticality_returns_correct_values(self):
        """Test if  get_criticality returns a Series with correct values"""

        n = 11
        anomalies = pd.Series([True, False, True, True, True, True, True, False, False, False, False],
                              index=pd.date_range('2022-01-01', freq='D', periods=n))
        normal_idx = pd.Series([False, True, True, True, True, False, False, True, True, True, True],
                               index=pd.date_range('2022-01-01', freq='D', periods=n))
        result = analysis.calculate_criticality(anomalies, normal_idx)
        expected_values = [0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0]
        expected_result = pd.Series(expected_values, index=pd.date_range('2022-01-01', freq='D', periods=n))
        pd.testing.assert_series_equal(result, expected_result)

    def test_calculate_arcana_importances(self):
        data = {
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [0.4, 0.5, 0.6],
            'feature3': [0.7, 0.8, 0.9]
        }
        bias_data = pd.DataFrame(data, index=pd.date_range("2023-01-01", periods=3))

        # Sample normal_index
        normal_index = pd.Series(index=bias_data.index, data=[True, False, True])

        # Test without start and end
        importances = calculate_mean_arcana_importances(bias_data)
        relative_importances = bias_data.abs()
        sums = bias_data.abs().sum(axis=1)
        for i, sum_value in enumerate(sums):
            relative_importances.iloc[i] /= sum_value
        expected_importances = relative_importances.mean(axis=0).sort_values(ascending=True)

        pd.testing.assert_series_equal(importances, expected_importances)

        # Test with start and end
        importances = calculate_mean_arcana_importances(bias_data, start="2023-01-01", end="2023-01-02")
        expected_importances = relative_importances.loc["2023-01-01":"2023-01-02"].mean(axis=0).sort_values(
            ascending=True)
        pd.testing.assert_series_equal(importances, expected_importances)
