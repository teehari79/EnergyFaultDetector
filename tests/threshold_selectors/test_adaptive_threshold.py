from unittest import TestCase

from datetime import datetime
import numpy as np
import pandas as pd
import keras
from numpy.testing import assert_array_equal

from energy_fault_detector.anomaly_scores.rmse_score import RMSEScore
from energy_fault_detector.threshold_selectors.adaptive_threshold import AdaptiveThresholdSelector


class TestAdaptiveThresholdSelector(TestCase):
    def setUp(self) -> None:
        # Set the seed using keras.utils.set_random_seed. This will set:
        # 1) `numpy` seed
        # 2) backend random seed
        # 3) `python` random seed
        keras.utils.set_random_seed(42)
        self.threshold_selector = AdaptiveThresholdSelector(gamma=1., nn_size=10, nn_epochs=100, early_stopping=True,
                                                            patience=3, validation_split=0.25)

        # input
        train_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        pred_data = np.array(np.arange(1, 100).reshape(33, 3) / 100)
        train_timestamps = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 2), periods=len(train_data))
        pred_timestamps = pd.date_range(start=datetime(2020, 1, 2), end=datetime(2020, 1, 3), periods=len(pred_data))
        self.train_data = pd.DataFrame(data=train_data, index=train_timestamps)
        train_normal_index = np.array(33 * [True])
        self.train_normal_index = pd.Series(data=train_normal_index, index=train_timestamps)
        self.pred_data = pd.DataFrame(data=pred_data, index=pred_timestamps)
        pred_normal_index = np.array([False] * 4 + [True] * 25 + [False] * 4)
        self.pred_normal_index = pd.Series(data=pred_normal_index, index=pred_timestamps)
        self.pred_data[~self.pred_normal_index] += np.random.normal(loc=4, scale=2,
                                                                    size=self.pred_data[~self.pred_normal_index].shape)

        # rmse object needed for testing
        self.rmse = RMSEScore()
        self.rmse.fit(self.train_data)

    def test_fit(self) -> None:
        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(self.train_data, scores, self.train_normal_index)
        self.assertIsNotNone(self.threshold_selector.nn_model)
        self.fitted_adaptive_threshold_selector = self.threshold_selector

    def test_predict(self) -> None:
        # expected output
        exp_anomalies = [True] * 4 + [False] * 25 + [True] * 4

        scores = self.rmse.transform(self.train_data)
        self.threshold_selector.fit(scaled_ae_input=self.train_data,
                                    anomaly_score=scores,
                                    normal_index=self.train_normal_index
                                    )
        scores = self.rmse.transform(self.pred_data)
        anomalies, threshold = self.threshold_selector.predict(scaled_ae_input=self.pred_data, x=scores)

        assert_array_equal(anomalies, exp_anomalies)

    def test_smoothing(self):
        anomaly_scores = self.rmse.transform(self.pred_data)
        index, mean_score = self.threshold_selector._smooth_anomaly_score(anomaly_score=anomaly_scores)
        expected_index = pd.to_datetime(['2020-01-02 00:00:00', '2020-01-02 00:45:00',
                                         '2020-01-02 01:30:00', '2020-01-02 02:15:00',
                                         '2020-01-02 03:00:00', '2020-01-02 03:45:00',
                                         '2020-01-02 04:30:00', '2020-01-02 05:15:00',
                                         '2020-01-02 06:00:00', '2020-01-02 06:45:00',
                                         '2020-01-02 07:30:00', '2020-01-02 08:15:00',
                                         '2020-01-02 09:00:00', '2020-01-02 09:45:00',
                                         '2020-01-02 10:30:00', '2020-01-02 11:15:00',
                                         '2020-01-02 12:00:00', '2020-01-02 12:45:00',
                                         '2020-01-02 13:30:00', '2020-01-02 14:15:00',
                                         '2020-01-02 15:00:00', '2020-01-02 15:45:00',
                                         '2020-01-02 16:30:00', '2020-01-02 17:15:00',
                                         '2020-01-02 18:00:00', '2020-01-02 18:45:00',
                                         '2020-01-02 19:30:00', '2020-01-02 20:15:00',
                                         '2020-01-02 21:00:00', '2020-01-02 21:45:00',
                                         '2020-01-02 22:30:00', '2020-01-02 23:15:00',
                                         '2020-01-03 00:00:00'])
        expected_mean_score = pd.Series(data=np.array([14.86307, 15.97868, 17.91221, 12.197, 1.26025, 1.15523, 1.05021,
                                                       0.94519, 0.84017, 0.73515, 0.63013, 0.52511, 0.42008, 0.31506,
                                                       0.21004, 0.10502, 0.0, 0.10502, 0.21004, 0.31506, 0.42008,
                                                       0.52511, 0.63013, 0.73515, 0.84017, 0.94519, 1.05021, 1.15523,
                                                       1.26025, 10.09686, 13.11089, 16.17702, 12.81238]))
        pd.testing.assert_index_equal(left=index, right=expected_index)
        pd.testing.assert_series_equal(left=mean_score, right=expected_mean_score, atol=1e-4, check_index=False)
