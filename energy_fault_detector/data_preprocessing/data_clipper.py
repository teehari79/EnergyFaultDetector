"""Clip data before standardization or normalization"""

import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')


# noinspection PyAttributeOutsideInit
class DataClipper(DataTransformer):
    """Clip data to remove outliers.

    Args:
        lower_percentile (float): The lower percentile for clipping (default: 0.01).
        upper_percentile (float): The upper percentile for clipping (default: 0.99).
        features_to_exclude (List[str]): A list of column names representing feature that should not be clipped.

    """

    def __init__(self, lower_percentile: float = 0.01, upper_percentile: float = 0.99,
                 features_to_exclude: List[str] = None):
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.feature_to_exclude: List[str] = features_to_exclude if features_to_exclude is not None else []

    def fit(self, x: Union[np.array, pd.DataFrame], y: Optional[np.array] = None) -> 'DataClipper':
        """Set feature names in and out."""
        self.feature_names_in_ = x.columns.to_list()
        self.feature_names_out_ = x.columns.to_list()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Clips the data to remove outliers, excluding angles.

        Args:
            x (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The clipped DataFrame.
        """

        check_is_fitted(self)
        # Exclude columns representing angles
        x_ = x.copy()
        x_without_feature_to_exclude = x_[[col for col in x_.columns if col not in self.feature_to_exclude]]
        # Exclude non-numeric columns
        x_numeric = x_without_feature_to_exclude.select_dtypes(include=np.number)

        # Clip the data using the specified percentiles
        x_clipped = x_numeric.clip(
            x_numeric.quantile(self.lower_percentile),
            x_numeric.quantile(self.upper_percentile),
            axis=1
        )

        # Update the original DataFrame with the clipped values
        x_[x_clipped.columns] = x_clipped

        return x_

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Not implemented for data clipper (not useful)"""
        return x

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names in the output."""
        check_is_fitted(self)
        return self.feature_names_out_
