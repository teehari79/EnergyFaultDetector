
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.anomaly_score import AnomalyScore

DataType = Union[pd.DataFrame, np.ndarray]


class RMSEScore(AnomalyScore):
    """Calculate the RMSE of given reconstruction errors.

    Attributes:
        scale: If True, mean and std of the training/fit reconstruction errors will be used to standardize recon errors
            during transform. Default: True

    Configuration example:

    .. code-block:: text

        train:
          anomaly_score:
            name: rmse
            params:
              scale: false

    """

    def __init__(self, scale: bool = True, **kwargs):

        super().__init__(**kwargs)
        self.scale = scale

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: DataType, y: Optional[pd.Series] = None) -> 'RMSEScore':
        """Calculate standard deviation and mean on training data

        Args:
            x: numpy 2d array with differences between prediction and actual sensor values
            y (optional): not used, labels indicating whether sample is normal (True) or anomalous (False).
        """
        if not hasattr(self, 'scale'):
            # backwards compatibility, add missing attribute and set to True (as that was the standard behaviour)
            self.scale = True

        if self.scale:
            # fitted attributes need trailing underscore - and are not initialized
            data = self._to_numpy(x)

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_samples = data.shape[0]
            if n_samples == 0:
                raise ValueError("Cannot fit RMSEScore on empty data.")

            # Calculate mean and standard deviation without allocating an
            # additional array the size of the dataset.
            self.mean_x_: np.ndarray = np.sum(data, axis=0) / n_samples
            sum_squared = np.einsum('ij,ij->j', data, data, optimize=True)
            variance = sum_squared / n_samples - np.square(self.mean_x_)
            variance = np.maximum(variance, 0.0)
            self.std_x_: np.ndarray = np.sqrt(variance)

        self.fitted_ = True  # nothing to fit
        return self

    def transform(self, x: DataType) -> pd.Series:
        """Calculate the RMSE based on the deviation matrix.

        Args:
            x: numpy 2d array or pandas Dataframe with differences between prediction and actual sensor values

        Returns:
            RMSE for each sample.
        """
        if not hasattr(self, 'scale'):
            # backwards compatibility, add missing attribute and set to True (as that was the standard behaviour)
            self.scale = True

        check_is_fitted(self)

        original_index = x.index if isinstance(x, pd.DataFrame) else None
        data = self._to_numpy(x)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if self.scale:
            # standardization of the reconstruction error in X
            if np.all(self.std_x_ > 0):
                data = (data - self.mean_x_) / self.std_x_
            else:
                data = data - self.mean_x_
            # replace possible inf values with 0
            data[np.isinf(data)] = 0

        if data.shape[1] == 0:
            raise ValueError("Input data must contain at least one feature.")

        mean_squared = np.einsum('ij,ij->i', data, data, optimize=True) / data.shape[1]
        scores = np.sqrt(mean_squared)
        if original_index is not None:
            scores = pd.Series(scores, index=original_index)

        return scores

    @staticmethod
    def _to_numpy(x: DataType) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=np.float64, copy=False)
        if isinstance(x, pd.Series):
            return x.to_numpy(dtype=np.float64, copy=False)
        return np.asarray(x, dtype=np.float64)
