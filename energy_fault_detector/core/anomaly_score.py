
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin

DataType = Union[pd.DataFrame, np.ndarray]


class AnomalyScore(BaseEstimator, TransformerMixin, SaveLoadMixin, ABC):
    """Anomaly score template"""

    @abstractmethod
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'AnomalyScore':
        """Fit the scorer.

        Args:
            x: numpy 2d array with differences between prediction and actual sensor values
            y: labels indicating whether sample is normal (True) or anomalous (False).
        """

    @abstractmethod
    def transform(self, x: DataType) -> pd.Series:
        """Implement transform method.

        Args:
            x: input data

        Returns:
            Scores
        """
