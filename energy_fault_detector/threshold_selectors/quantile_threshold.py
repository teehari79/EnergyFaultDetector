
from typing import Union

import numpy as np
import pandas as pd

from energy_fault_detector.core.threshold_selector import ThresholdSelector

Array1D = Union[np.ndarray, pd.Series]


class QuantileThresholdSelector(ThresholdSelector):
    """Find a threshold by defining a specified quantile of the given anomaly scores.

    Args:
        quantile (float): The quantile of the scores to be computed. Defaults to 0.99.

    Attributes:
        threshold (float): Scores above the threshold are classified as anomalies, while scores below are classified as
            normal.

    Example Configuration:

    .. code-block:: text

        train:
          threshold_selector:
            name: QuantileThresholdSelector
            params:
              quantile: 0.99
    """

    def __init__(self, quantile: float = 0.99):
        super().__init__()

        self.quantile = quantile

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: Array1D, y: pd.Series = None) -> 'QuantileThresholdSelector':
        """
        Sets the threshold to the chosen quantile of the provided anomaly scores.

        Args:
            x (Array1D): Array containing calculated anomaly scores.
            y (pd.Series, optional): Labels indicating whether each sample is normal (True) or anomalous (False).
                Optional; if not provided, it is assumed that all data represents normal behavior.

        Returns:
            QuantileThresholdSelector: The instance of this class after setting the threshold.

        Raises:
            Warning: If a suitable threshold cannot be found, the threshold is set to the maximum score.
        """

        if isinstance(x, pd.Series):
            x_series = x.sort_index()
        else:
            x_series = pd.Series(np.asarray(x))

        selected_scores: Union[pd.Series, np.ndarray]

        if y is not None:
            if isinstance(y, pd.DataFrame):
                if y.shape[1] != 1:
                    raise ValueError('QuantileThresholdSelector requires a one-dimensional normal index.')
                y_series = y.iloc[:, 0]
            elif isinstance(y, pd.Series):
                y_series = y
            else:
                y_array = np.asarray(y)
                if y_array.ndim != 1:
                    raise ValueError('QuantileThresholdSelector requires a one-dimensional normal index.')
                if y_array.shape[0] != len(x_series):
                    raise ValueError('Boolean index must be the same length as the anomaly scores.')
                selected_scores = x_series[y_array.astype(bool)]
                y_series = None

            if y is not None and isinstance(y, (pd.Series, pd.DataFrame)):
                y_series = y_series.sort_index()
                if isinstance(x, pd.Series):
                    x_series, y_series = x_series.align(y_series, join='inner')
                else:
                    if not y_series.index.equals(x_series.index):
                        y_series = pd.Series(y_series.to_numpy(), index=x_series.index)
                y_mask = y_series.fillna(False).astype(bool)
                if len(y_mask) != len(x_series):
                    raise ValueError('Boolean index must be the same length as the anomaly scores.')
                selected_scores = x_series[y_mask]

            if isinstance(selected_scores, pd.Series) and selected_scores.empty:
                selected_scores = x_series
        else:
            selected_scores = x_series

        if isinstance(selected_scores, pd.Series):
            x_ = selected_scores.to_numpy(dtype=np.float64, copy=False)
        else:
            x_ = np.asarray(selected_scores, dtype=np.float64)

        if x_.size == 0:
            x_ = x_series.to_numpy(dtype=np.float64, copy=False)

        self.threshold = float(np.quantile(x_, self.quantile))

        if self.threshold is None:
            self.threshold = float(np.sort(x)[-1])
            raise Warning('Could not find suitable threshold, `threshold` is set to max score.')
        return self
