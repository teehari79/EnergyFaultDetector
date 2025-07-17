from typing import Union, Optional

import numpy as np
import pandas as pd

from energy_fault_detector.core.threshold_selector import ThresholdSelector

from .telemanon_errors import Errors

Array1D = Union[np.ndarray, pd.Series]


class TelemanomThreshold(ThresholdSelector):
    """Threshold selection method based on the non-parametric method from the paper
    'Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding'
    https://doi.org/10.1145/3219819.3219845

    Args:
        batch_size (int): Batch size.
        window_size (int): Window size, number of batches to consider.
        smoothing_perc (float): Smoothing percentage for the errors.
        l_s (int): Data points to ignore at the start.
        error_buffer (int): Number of values surrounding an anomaly, that are brought into the sequence (considered as
            possibly anomalous).
        p (float): Minimum percent decrease (as fraction) between max errors in anomalous sequences (for pruning).
            The difference between the max error below the threshold and above the threshold should be at least p.

    """

    def __init__(self, window_size: int = 30, batch_size: int = 70, smoothing_perc: float = 0.05, l_s: int = 10,
                 error_buffer: int = 100, p: float = 0.13):

        super().__init__()

        self.window_size = window_size
        self.batch_size = batch_size
        self.smoothing_perc = smoothing_perc
        self.l_s = l_s
        self.error_buffer = error_buffer
        self.p: float = p

        self.threshold: Optional[np.ndarray] = None
        self.smoothed_errors: Optional[np.ndarray] = None

    def __sklearn_is_fitted__(self):
        """The method does not need to be fitted."""
        return True

    def fit(self, x: Array1D, y: Array1D = None) -> 'TelemanomThreshold':
        """Does nothing, the threshold depends on the context. Just sets an initial threshold.

        Args:
            x: array with calculated anomaly scores
            y: true label (not used)

        Returns:
            The threshold selector.
        """

        self.threshold = x.mean() + 3 * x.std()
        return self

    def predict(self, x: Array1D, **kwargs) -> Array1D:
        """Return a boolean array indicating whether sample is anomalous based on the dynamic threshold.

        Args:
            x: array with calculated anomaly scores

        Returns:
            Boolean array where true = anomaly, false = normal.
        """

        telemanon_errors = Errors(
            prediction_error=x,
            window_size=self.window_size,
            batch_size=self.batch_size,
            smoothing_perc=self.smoothing_perc,
            l_s=self.l_s,
            error_buffer=self.error_buffer,
            p=self.p,
        )

        telemanon_errors.process_batches()
        threshold = telemanon_errors.threshold
        if len(threshold) < len(x):
            # add last threshold difference in length times
            last_threshold_value = threshold[-1]
            additional_length = len(x) - len(threshold)
            threshold = np.concatenate([threshold, np.full(additional_length, last_threshold_value)])

        if isinstance(x, pd.Series):
            threshold = pd.Series(threshold, index=x.index)

        self.threshold = threshold
        self.smoothed_errors = telemanon_errors.e_s

        e_index = np.arange(1, len(x) + 1)
        if isinstance(x, pd.Series):
            return pd.Series([idx in telemanon_errors.i_anom for idx in e_index], index=x.index)

        return np.array([idx in telemanon_errors.i_anom for idx in e_index])
