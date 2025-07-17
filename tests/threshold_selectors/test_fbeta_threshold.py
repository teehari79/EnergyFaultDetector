
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.metrics import fbeta_score

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore
from energy_fault_detector.threshold_selectors.fbeta_threshold import FbetaSelector


class TestFbetaSelector(TestCase):
    def setUp(self) -> None:
        self.threshold_selector = FbetaSelector(beta=0.5)

        # input
        self.train_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        self.normal_index = pd.Series([False] * 4 + [True] * 25 + [False] * 4)

        # rmse object needed for testing
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

    def test_fit(self) -> None:
        # expected score
        exp_score = 1.0

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)

        fbeta = fbeta_score(~self.normal_index,
                            scores > self.threshold_selector.threshold,
                            beta=self.threshold_selector.beta)
        assert_array_equal(exp_score, fbeta)

    def test_predict(self) -> None:
        # expected output
        exp_anomalies = [True]*4 + [False]*25 + [True]*4

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scores, self.normal_index)
        anomalies = self.threshold_selector.predict(scores)

        assert_array_equal(anomalies, exp_anomalies)
