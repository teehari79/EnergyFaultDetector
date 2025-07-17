
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin

Array1D = Union[np.ndarray, pd.Series]


class ThresholdSelector(BaseEstimator, ClassifierMixin, SaveLoadMixin):
    """Template for threshold selectors.

    Finds the threshold (fit method) of the given reconstruction errors (x), to be considered anomalous.
    The predict method returns an array of boolean, indicating which samples of the input are anomalous.

    Attributes:
        threshold: scores above the threshold is classified as anomaly, below is classified as normal.
    """

    def __init__(self):
        super().__init__()
        self.threshold = None

    def fit(self, x: Array1D, y: Array1D = None) -> 'ThresholdSelector':  # pylint: disable=W0237
        """Fit ThresholdSelector object on anomaly score values x and
        labels y to determine the score threshold. For prediction, every sample with
        a scores above this threshold is considered an anomaly.

        Args:
            x: array or pandas Series with calculated anomaly scores
            y: labels indicating whether sample is normal (True) or anomalous (False)
                Optional, if not given, we assume all data provided represents normal behaviour

        Returns:
            The threshold selector
        """
        raise NotImplementedError('The fit methods needs to be implemented.')

    def predict(self, x: Array1D, **kwargs) -> Union[Array1D, Tuple[Array1D, Array1D]]:
        """Return a boolean array indicating whether sample is anomalous.

        Args:
            x: array with calculated anomaly scores
            kwargs: other arguments for specific threshold selectors.

        Returns:
            Boolean array where true = anomaly, false = normal
        """

        check_is_fitted(self)
        # noinspection PyUnresolvedReferences
        # pylint: disable=no-member
        return x > self.threshold

    def __sklearn_is_fitted__(self):
        """Needed for check_is_fitted"""
        return self.threshold is not None
