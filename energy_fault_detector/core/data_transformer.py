
from typing import List
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin


class DataTransformer(BaseEstimator, TransformerMixin, SaveLoadMixin, ABC):
    """DataTransformer template."""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> 'DataTransformer':
        """Fit the preprocessor with training data. Should set attributes `feature_names_in_` and `n_features_in_`.
        If columns are selected/dropped, should also set `feature_names_out_` and `columns_dropped_`.

        Args:
            x: pandas dataframe with input data
            y: (optional) labels indicating whether sample is normal (True) or anomalous (False).
        """

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Implement transform method.

        Args:
            x: input data.
        """

    @abstractmethod
    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Implement inverse transform method

        Args:
            x: input data.
        """

    @abstractmethod
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation.

        Returns
            feature_names_out : list/np array of str objects.
        """
