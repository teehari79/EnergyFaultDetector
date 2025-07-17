
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore


class TestRMSEScore(TestCase):
    def setUp(self) -> None:
        self.rmse_score = RMSEScore()
        self.rmse_score_new = RMSEScore(scale=False)
        self.train_data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.test_data = np.array([
            [1, 5, 6],
            [4, 8, 6],
        ])
        self.rmse_score_new.fit(self.train_data)  # does nothing

    def test_fit(self) -> None:

        self.rmse_score.fit(self.train_data)
        assert_array_equal(np.array([4., 5., 6.]), self.rmse_score.mean_x_)
        assert_array_almost_equal(np.array([2.44948974, 2.44948974, 2.44948974]), self.rmse_score.std_x_)

        self.assertTrue(self.rmse_score_new.fitted_)

    def test_transform(self) -> None:
        self.rmse_score.fit(self.train_data)
        score = self.rmse_score.transform(self.test_data)
        assert_array_almost_equal(np.array([0.70710678, 0.70710678]), score)

        score_new = self.rmse_score_new.transform(self.test_data)
        score_new_expected = np.sqrt(np.mean(self.test_data**2, axis=1))
        assert_array_almost_equal(score_new, score_new_expected)

    def test_transform_not_fitted(self) -> None:
        with self.assertRaises(ValueError):
            self.rmse_score.transform(self.test_data)
