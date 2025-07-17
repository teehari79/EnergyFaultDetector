
from typing import Optional, Dict, Tuple, List, Any, Union
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

from energy_fault_detector.utils.analysis import calculate_criticality

logger = logging.getLogger('energy_fault_detector')


class CAREScore:
    """Calculate CARE Score (coverage, accuracy, reliability, and earliness) for an anomaly detection algorithm, as
    described in the paper 'CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine
    Data' (https://doi.org/10.3390/data9120138).

    For each normal and anomalous event in your dataset call `evaluate_event`. Afterwards, call `get_final_score`
    to calculate the CARE score of your model.

    Coverage is measured by the pointwise F-Score of anomalous events, the accuracy measured as the accuracy of normal
    events, the reliability by the eventwise F-score over all events and the earliness as weighted score over anomalous
    events.

    The method `evaluate_event` calculates the coverage, accuracy and weighted score for a specific event, as well as
    whether the event was detected as anomaly or not. It collects this information in a dataframe.
    Finally, the `get_final_score` is called to calculate the CARE score over all (or a part of the) evaluated events.

    Args:
        coverage_beta (float): Beta parameter for coverage F-score calculation. Default is 1.0.
        eventwise_f_score_beta (float): Beta parameter for event-wise F-score calculation. Default is 1.0.
        coverage_w (float): Weight for coverage in the final CARE score calculation. Default is 1.0.
        accuracy_w (float): Weight for accuracy in the final CARE score calculation. Default is 2.0.
        weighted_score_w (float): Weight for the weighted score in the final CARE score calculation. Default is 1.0.
        eventwise_f_score_w (float): Weight for the event-wise F-score in the final CARE score calculation.
            Default is 1.0.
        anomaly_detection_method (str): Method used to calculate anomaly detection score. Either criticality or
            fraction. Default is 'criticality'. If fraction is used, an event is detected as anomaly if at least
            `min_fraction_anomalous_timestamps` of the 'normal' timestamps are detected as anomaly. If criticality is
            used, an event is detected as anomaly if the maximum criticality exceed criticality_threshold.
        criticality_threshold (int): Threshold for criticality. If criticality exceeds this threshold, the event will be
            counted as an anomaly event (if anomaly_detection_method = criticality). Default is 72.
        min_fraction_anomalous_timestamps (float): Minimum fraction of anomalous timestamps to consider an event
            detected (if anomaly_detection_method = fraction). Default is 0.1.
        ws_start_of_descend (Tuple[int, int]): Must be a fraction (tuple with numerator and denominator) between 0
            and 1. It determines the point after which the scoring weights for the weighted score decrease.
            Default is (1, 4).
    """

    def __init__(self, coverage_beta: float = 0.5, eventwise_f_score_beta: float = 0.5, coverage_w: float = 1.,
                 accuracy_w: float = 2., weighted_score_w: float = 1., eventwise_f_score_w: float = 1.,
                 criticality_threshold: int = 72, min_fraction_anomalous_timestamps: float = 0.1,
                 ws_start_of_descend: Tuple[int, int] = (1, 4), anomaly_detection_method: str = 'criticality'):

        if anomaly_detection_method not in ['criticality', 'fraction']:
            raise ValueError("Anomaly detection method must be either 'criticality' or 'fraction'")

        self.coverage_beta = coverage_beta
        self.eventwise_f_score_beta = eventwise_f_score_beta

        self.coverage_w = coverage_w
        self.accuracy_w = accuracy_w
        self.weighted_score_w = weighted_score_w
        self.eventwise_f_score_w = eventwise_f_score_w

        self.min_fraction_anomalous_timestamps = min_fraction_anomalous_timestamps
        self.criticality_threshold = criticality_threshold
        self.anomaly_detection_method = anomaly_detection_method

        self.ws_start_of_descend = ws_start_of_descend

        self._evaluated_events: List[Dict[str, Any]] = []

    @property
    def evaluated_events(self) -> pd.DataFrame:
        """Pandas DataFrame with evaluated events."""
        return pd.DataFrame(self._evaluated_events)

    def evaluate_event(self, event_start: Union[int, pd.Timestamp], event_end: Union[int, pd.Timestamp],
                       event_label: str,
                       normal_index: pd.Series, predicted_anomalies: pd.Series,
                       event_id: int = None, ignore_normal_index: bool = False) -> Dict[str, float]:
        """Evaluate the performance of an anomaly detection model for a given event.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            event_label (str): True label of the event. This can be either 'anomaly' or 'normal'.
            normal_index (pd.Series): Boolean mask indicating normal timestamps. ID int as index.
            predicted_anomalies (pd.Series): Boolean pandas series, indicating an anomaly was detected. ID int as index.
            event_id (int): ID of event. If not specified, a counter is used instead.
            ignore_normal_index (bool): Whether to ignore the normal index and evaluate all timestamps in the prediction
                or test dataset. Default False.

        Returns:
            Dict[str, float]: Dictionary containing the calculated metrics.
        """

        if event_label not in ['anomaly', 'normal']:
            raise ValueError('Unknown event label (should be either `anomaly` or `normal`')

        normal_index = normal_index.sort_index()
        if ignore_normal_index:
            normal_index = pd.Series([True] * len(normal_index), index=normal_index.index)
        predicted_anomalies = predicted_anomalies.sort_index()

        ground_truth = self.create_ground_truth(event_start, event_end, normal_index, event_label)
        predicted_anomalies_event = predicted_anomalies.loc[event_start:event_end]
        normal_index_event = normal_index.loc[event_start:event_end]

        if event_label == 'anomaly':
            weighted_score_value = self._calculate_weighted_score(event_prediction=predicted_anomalies_event)
        else:
            weighted_score_value = np.nan

        n_normal_timestamps_event = len(predicted_anomalies_event[normal_index_event])
        n_anomalies_detected_normal_only_event = predicted_anomalies_event[normal_index_event].sum()
        anomaly_event_detected = (
                n_anomalies_detected_normal_only_event / n_normal_timestamps_event
                >= self.min_fraction_anomalous_timestamps
        )

        max_criticality = np.max(calculate_criticality(anomalies=predicted_anomalies, normal_idx=normal_index))

        # we only evaluate predicted anomalies during expected normal operation
        metrics = self._calculate_metrics(
            ground_truth=ground_truth[normal_index],
            predicted_anomalies=predicted_anomalies[normal_index],
            event_label=event_label
        )

        if event_id is None:
            event_id = len(self._evaluated_events)

        evaluation = {
            'event_id': event_id,
            'event_label': event_label,  # true label
            'anomaly_event_detected_by_fraction': anomaly_event_detected,  # predicted label by fraction
            'anomaly_detected_by_criticality': max_criticality > self.criticality_threshold,  # predicted label
            'weighted_score': weighted_score_value,
            'max_criticality': max_criticality,
            **metrics  # F-score, Accuracy, TP, TN, FN, FP
        }

        # Add evaluation to dataframe
        self._evaluated_events.append(evaluation)

        return evaluation

    def get_final_score(self, event_selection: Optional[List[int]] = None) -> float:
        """Calculate the CARE score over all events in self.evaluated_events or a selection of the events.

        If the average accuracy over all normal events < 0.5, CARE score = average accuracy over all normal events
            (worse than random guessing).
        If no anomalies were detected, the CARE score = 0.
        Else, the CARE score is calculated as:

            ( (average F-score over all anomaly events) * coverage_w
             + (average weighted score over all anomaly events) * weighted_score_w
             + (average accuracy over all normal events) * accuracy_w
             + event wise F-score * eventwise_f_score_w ) / sum_of_weights

        where `sum_of_weights` = coverage_w + weighted_score_w + accuracy_w + eventwise_f_score_w.

        Returns:
            float: CARE score
        """

        anomaly_detected = ('anomaly_detected_by_criticality' if self.anomaly_detection_method == 'criticality'
                            else 'anomaly_detected_by_fraction')

        if event_selection is None:
            events_to_evaluate = self.evaluated_events
        else:
            events_to_evaluate = self.evaluated_events[self.evaluated_events['event_id'].isin(event_selection)]

        if np.sum(events_to_evaluate[anomaly_detected]) == 0:
            logger.info('No anomalies were detected')
            return 0.

        is_anomaly_event = events_to_evaluate['event_label'] == 'anomaly'

        avg_accuracy = events_to_evaluate.loc[~is_anomaly_event, 'accuracy'].mean()
        if avg_accuracy <= 0.5:
            logger.info('Accuracy over all normal events <0.5')
            return avg_accuracy

        avg_f_score = events_to_evaluate.loc[is_anomaly_event, 'f_beta_score'].mean()
        avg_weighted_score = events_to_evaluate.loc[is_anomaly_event, 'weighted_score'].mean()
        eventwise_fscore = fbeta_score(
            y_true=is_anomaly_event,
            y_pred=events_to_evaluate[anomaly_detected],
            beta=self.eventwise_f_score_beta
        )

        care_score = (avg_accuracy * self.accuracy_w
                      + avg_weighted_score * self.weighted_score_w
                      + avg_f_score * self.coverage_w
                      + eventwise_fscore * self.eventwise_f_score_w)
        sum_of_weights = (self.accuracy_w + self.weighted_score_w +
                          self.coverage_w + self.eventwise_f_score_w)
        care_score /= sum_of_weights

        return care_score

    @staticmethod
    def create_ground_truth(event_start: Union[int, pd.Timestamp], event_end: Union[int, pd.Timestamp],
                            normal_index: pd.Series, event_label: str) -> pd.Series:
        """Create the ground truth labels based on the provided inputs.

        Args:
            event_start (int, pd.Timestamp): Start index/timestamp of the event.
            event_end (int, pd.Timestamp): End index/timestamp of the event.
            normal_index (pd.Series): Boolean mask indicating normal samples.
            event_label (str): True label indicating the type of the event (anomaly or normal).

        Returns:
            pd.Series: Ground truth labels. True if the timestamp is an anomaly, False otherwise.
        """

        ground_truth = pd.Series(data=~normal_index, index=normal_index.index, name='ground_truth')
        ground_truth = ground_truth.sort_index()
        if event_label == 'anomaly':
            ground_truth.loc[event_start:event_end] = True
        return ground_truth

    def _calculate_metrics(self, ground_truth: pd.Series, predicted_anomalies: pd.Series,
                           event_label: str) -> Dict[str, float]:
        """Calculate various metrics for the given labels and scores.

        Args:
            ground_truth (pd.Series): Ground truth labels. Timestamp as index.
            predicted_anomalies (pd.Series): Predicted labels. Timestamp as index.
            event_label (str): normal or anomaly, indicates whether F-beta-Score can be calculated.

        Returns:
            Dict[str, float]: Dictionary containing the calculated metrics.
        """

        f_score = np.nan
        if event_label == 'anomaly':
            f_score = fbeta_score(y_true=ground_truth, y_pred=predicted_anomalies, beta=self.coverage_beta)

        accuracy = accuracy_score(y_true=ground_truth, y_pred=predicted_anomalies)
        tn, fp, fn, tp = confusion_matrix(y_true=ground_truth, y_pred=predicted_anomalies, labels=[False, True]).ravel()

        return {
            'f_beta_score': f_score,
            'accuracy': accuracy,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        }

    def _calculate_weighted_score(self, event_prediction: pd.Series) -> float:
        """Calculate the weighted score based on a modification of the linear weighting function.

        For each element of event_prediction, this function computes a weight between 0 and 1 which is then multiplied
        with the anomaly prediction of the element. In the end, the weighted score is the normalized sum of weights *
        event_prediction.

        Args:
            event_prediction (pd.Series): Boolean model prediction during the event. True = Anomaly and False = Normal.

        Returns:
            float: Weighted score for the event; higher means earlier detection.
        """

        event_length = len(event_prediction)

        start_of_descend_numerator, start_of_descend_denominator = self.ws_start_of_descend
        start_of_descend_float = start_of_descend_numerator / start_of_descend_denominator

        scale = start_of_descend_denominator
        scaled_event_length = event_length * scale
        cp = int(scaled_event_length * start_of_descend_float)

        x_values = np.linspace(0, scale, scaled_event_length)
        weights = np.zeros(scaled_event_length)
        weights[:cp] = scale
        slope = 1 / (1 - start_of_descend_float)
        offset = scale / (1 - start_of_descend_float)
        weights[cp:scaled_event_length] = offset - slope * x_values[cp:scaled_event_length]

        final_weights = weights[::scale]
        weights = final_weights / scale

        return np.sum(event_prediction * weights) / np.sum(weights)

    def save_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Save the evaluated events to a CSV file.

        Args:
            file_path (Path): The file path where the evaluated events will be saved.
        """
        self.evaluated_events.to_csv(Path(file_path), index=False)

    def load_evaluated_events(self, file_path: Union[Path, str]) -> None:
        """Load evaluated events from a CSV file.

        Args:
            file_path (Path): The file path from which the evaluated events will be loaded.
        """
        file_path = Path(file_path)
        if file_path.exists():
            self._evaluated_events = pd.read_csv(file_path).to_dict(orient='records')
        else:
            raise FileNotFoundError(f"File {file_path} does not exist.")
