"""Analysis utility functions"""

from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


DurationInput = Optional[Union[str, int, float, timedelta, pd.Timedelta, np.timedelta64]]


def _parse_duration(duration: DurationInput) -> Optional[pd.Timedelta]:
    """Convert supported duration specifications to :class:`pandas.Timedelta`.

    Args:
        duration: Duration value provided by the caller. Supported input types are strings accepted by
            :func:`pandas.to_timedelta`, numeric values interpreted as seconds and timedelta instances.

    Returns:
        A :class:`pandas.Timedelta` instance or ``None`` when ``duration`` is ``None``.
    """

    if duration in (None, ""):
        return None

    if isinstance(duration, pd.Timedelta):
        return duration

    if isinstance(duration, np.timedelta64):
        return pd.to_timedelta(duration)

    if isinstance(duration, timedelta):
        return pd.to_timedelta(duration)

    if isinstance(duration, (int, float)):
        return pd.to_timedelta(duration, unit="s")

    return pd.to_timedelta(duration)


def create_events(sensor_data: pd.DataFrame, boolean_information: pd.Series,
                  min_event_length: Optional[int] = 10,
                  min_event_duration: DurationInput = None
                  ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Create an event DataFrame based on boolean information such as predicted anomalies or a normal index
    and return a list of event DataFrames intended for further evaluation.

    Args:
        sensor_data (pd.DataFrame): A DataFrame with a timestamp as index and numerical sensor data.
        boolean_information (pd.Series): A Series with a timestamp as index and boolean values indicating events.
        min_event_length (Optional[int], optional): The smallest number of consecutive True timestamps needed to define
            a critical event. When ``None`` no minimum length constraint is applied. Defaults to 10.
        min_event_duration (optional): Minimum duration that consecutive anomalies must span to qualify as a critical
            event. Strings supported by :func:`pandas.to_timedelta`, numeric values interpreted as seconds and
            ``timedelta`` objects are accepted. When ``None`` the duration criterion is ignored.

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame]]: A tuple containing:
            - event_meta_data (pd.DataFrame): A DataFrame with columns 'start', 'end', and 'duration' for each event.
            - event_data (List[pd.DataFrame]): A list of DataFrames corresponding to the sensor data during the defined events.
    """
    # Ensure the boolean information uses the same index as the sensor data. When running predictions on
    # different assets we encountered cases where the boolean series used a different index. Pandas silently
    # reindexes boolean masks during ``DataFrame.__getitem__`` which, in this case, resulted in out of bounds
    # indices and raised an ``IndexError``. Aligning the series to the sensor data index avoids the
    # misalignment and guarantees a pure boolean mask of equal length.


    # 
    # print("boolean_information index:",boolean_information,boolean_information.index)
    # print("min_event_length:",min_event_length)
    # print("sensor_data index:",sensor_data,sensor_data.index)
    sensor_data = sensor_data.groupby(level=0).mean()
    if boolean_information.index.has_duplicates:
        # ``Series.reindex`` does not support duplicate indices. Duplicate timestamps can appear when
        # predictions are generated from concatenated batches. In this context we treat any duplicate
        # timestamp as an event whenever at least one of the values is ``True``.
        boolean_information = boolean_information.groupby(level=0).max()
    boolean_information = boolean_information.reindex(sensor_data.index, fill_value=False)

    # Create a boolean mask for consecutive True values
    mask = (boolean_information != boolean_information.shift()).cumsum()

    # Group by the mask and filter groups where bool_series is True and has more
    # than consecutive_true_value_threshold consecutive True
    bool_mask = boolean_information.groupby(mask).transform(lambda data: data.all()).fillna(False)
    grouped_sensor_data = sensor_data[bool_mask].groupby(mask)
    event_candidates = [group[1] for group in grouped_sensor_data]

    duration_threshold = _parse_duration(min_event_duration)

    event_meta_data = pd.DataFrame(columns=["start", "end", "duration"])
    if not event_candidates:
        return event_meta_data, []

    starts = [event.index[0] for event in event_candidates]
    ends = [event.index[-1] for event in event_candidates]
    durations = [end - start for start, end in zip(starts, ends)]

    event_meta_data = pd.DataFrame({
        'start': starts,
        'end': ends,
        'duration': durations,
    })

    def _satisfies_length(event_len: int) -> bool:
        if min_event_length is None:
            return False
        return event_len >= min_event_length

    def _satisfies_duration(event_duration: pd.Timedelta) -> bool:
        if duration_threshold is None:
            return False
        try:
            return pd.to_timedelta(event_duration) >= duration_threshold
        except (TypeError, ValueError):
            return False

    selected_events: List[pd.DataFrame] = []
    keep_mask: List[bool] = []
    for event, duration in zip(event_candidates, event_meta_data['duration']):
        length_ok = _satisfies_length(len(event))
        duration_ok = _satisfies_duration(duration)

        if duration_threshold is None and min_event_length is None:
            keep = True
        elif duration_threshold is None:
            keep = length_ok
        elif min_event_length is None:
            keep = duration_ok
        else:
            keep = length_ok or duration_ok

        keep_mask.append(keep)
        if keep:
            selected_events.append(event)

    if not selected_events:
        return pd.DataFrame(columns=['start', 'end', 'duration']), []

    event_meta_data = event_meta_data.loc[keep_mask].reset_index(drop=True)

    try:
        event_meta_data['start'] = event_meta_data['start'].dt.round('min')
        event_meta_data['end'] = event_meta_data['end'].dt.round('min')
    except AttributeError:
        # if index is not datetimelike an attribute error is thrown. In this case do nothing
        pass
    event_meta_data['duration'] = event_meta_data['end'] - event_meta_data['start']
    return event_meta_data, selected_events


def calculate_criticality(anomalies: pd.Series, normal_idx: pd.Series = None, init_criticality: int = 0,
                          max_criticality: int = 1000) -> pd.Series:
    """Calculate criticality based on anomaly detection results. Increases if an anomaly is detected during normal
    operation, eases if no anomalies are detected during normal operation. If normal_idx is not provided, it is assumed
    that all detected anomalies occur during normal operation.

    Args:
        anomalies (pd.Series): A pandas Series with boolean values indicating whether an anomaly was detected,
            indexed by timestamp.
        normal_idx (pd.Series, optional): A pandas Series with boolean values indicating normal operation, indexed by
            timestamp.
        init_criticality (int, optional): The initial criticality value. Defaults to 0.
        max_criticality (int, optional): The maximum criticality value. Defaults to 1000.

    Returns:
        pd.Series: A pandas Series representing the criticality over time, indexed by timestamp.

    Raises:
        ValueError: If the lengths of the given pandas Series for anomalies and normal_idx do not match.
    """

    if normal_idx is None:
        normal_idx = pd.Series(np.full(len(anomalies), True), index=anomalies.index)

    if len(anomalies) != len(normal_idx):
        raise ValueError('length of given pandas series anomalies and normal idx do not match!')

    anomalies = anomalies.sort_index()
    normal_idx = normal_idx.sort_index()
    criticality = [init_criticality]
    for i, anomaly in enumerate(anomalies):
        if normal_idx.iloc[i] and anomaly:
            # increase if anomaly detected during normal OP
            criticality.append(min(criticality[-1] + 1, max_criticality))
        elif normal_idx.iloc[i] and not anomaly:
            # decrease if no anomalies detected during normal OP
            criticality.append(max(criticality[-1] - 1, 0))
        else:
            criticality.append(criticality[-1])
    return pd.Series(criticality[1:], index=anomalies.index)
