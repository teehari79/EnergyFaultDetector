
from typing import Union
from datetime import datetime

import pandas as pd


def calculate_mean_arcana_importances(bias_data: pd.DataFrame, start: Union[str, datetime] = None,
                                      end: Union[str, datetime] = None) -> pd.Series:
    """ Calculate the mean ARCANA importances for a given period. If normal_index is provided, only the timestamps
    during normal status are considered.

    ARCANA importances express the contribution of each data feature to the resulting reconstruction error by comparing
    the provided bias data.

    Args:
        bias_data: pandas DataFrame containing ARCANA bias data, with timestamps as index.
        start: start of time period to evaluate.
        end: end of time period to evaluate.

    Returns:
        pandas Series with features/column names as index and importances as values.
    """

    bias_data_selection = bias_data
    if start is not None:
        bias_data_selection = bias_data.loc[start:end]

    importances = bias_data_selection.abs()
    sums = importances.sum(axis=1)
    for i, sum_value in enumerate(sums):
        importances.iloc[i] /= sum_value

    final_importances = importances.mean(axis=0).sort_values(ascending=True)
    return final_importances


def calculate_arcana_importance_time_series(bias_data: pd.DataFrame) -> pd.DataFrame:
    """ Calculate ARCANA importances for each time stamp.

    Args:
        bias_data: pandas DataFrame containing ARCANA bias data, with timestamps as index.

    Returns:
        pandas DataFrame with time stamps as index and importances for each feature as values.
    """

    importances = bias_data.abs()
    sums = importances.sum(axis=1)
    for i, sum_value in enumerate(sums):
        importances.iloc[i] /= sum_value
    return importances
