
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore
from energy_fault_detector.threshold_selectors.quantile_threshold import QuantileThresholdSelector


class TestQuantileSelector(TestCase):
    def setUp(self) -> None:
        self.threshold_selector = QuantileThresholdSelector(quantile=0.95)

        # input
        self.train_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        self.normal_index = np.array([False]*4 + [True]*25 + [False]*4)

        # rmse object needed for testing
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

    def test_fit(self) -> None:
        # expected output
        exp_threshold = 1.2392478743647883

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)

        assert_array_equal(exp_threshold, self.threshold_selector.threshold)

    def test_fit_without_label(self) -> None:
        # expected output
        exp_threshold_with_anomaly = 1.617323497052351

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores)

        assert_array_equal(exp_threshold_with_anomaly, self.threshold_selector.threshold)

    def test_predict(self) -> None:
        # expected output
        exp_anomalies = [True]*5 + [False]*23 + [True]*5

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)
        anomalies = self.threshold_selector.predict(scores)

        assert_array_equal(anomalies, exp_anomalies)
