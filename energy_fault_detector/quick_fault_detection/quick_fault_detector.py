"""Quick energy fault detection, to try out the EnergyFaultDetector model on a specific dataset."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
try:
    from typing import Literal
except ImportError:  # pragma: no cover - for Python<3.8 compatibility
    from typing_extensions import Literal

import pandas as pd

from energy_fault_detector._logs import setup_logging
from energy_fault_detector.config import Config
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.utils.analysis import create_events
from energy_fault_detector.root_cause_analysis.arcana_utils import calculate_mean_arcana_importances
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult, ModelMetadata

from energy_fault_detector.quick_fault_detection.data_loading import load_train_test_data
from energy_fault_detector.quick_fault_detection.configuration import select_config
from energy_fault_detector.quick_fault_detection.output import generate_output_plots, output_info

setup_logging(os.path.join(os.path.dirname(__file__), '..', 'logging.yaml'))
logger = logging.getLogger('energy_fault_detector')


def quick_fault_detector(
    csv_data_path: Optional[str],
    csv_test_data_path: Optional[str] = None,
    train_test_column_name: Optional[str] = None,
    train_test_mapping: Optional[dict] = None,
    time_column_name: Optional[str] = None,
    status_data_column_name: Optional[str] = None,
    status_mapping: Optional[dict] = None,
    status_label_confidence_percentage: Optional[float] = 0.95,
    features_to_exclude: Optional[List[str]] = None,
    angle_features: Optional[List[str]] = None,
    automatic_optimization: bool = True,
    enable_debug_plots: bool = False,
    min_anomaly_length: int = 18,
    critical_event_min_length: Optional[int] = None,
    critical_event_min_duration: Optional[Union[str, float, int]] = None,
    save_dir: Optional[str] = None,
    mode: Literal['train', 'predict'] = 'train',
    model_path: Optional[str] = None,
    model_directory: Optional[str] = None,
    model_subdir: Optional[str] = None,
    model_name: Optional[str] = None,
    asset_name: Optional[str] = None,
    rca_ignore_features: Optional[List[str]] = None,
    overwrite_models: bool = False,
) -> Tuple[
    FaultDetectionResult,
    pd.DataFrame,
    List[Dict[str, Any]],
    Optional[ModelMetadata],
]:
    """Analyzes provided data using an autoencoder based approach for identifying anomalies based on a learned normal
    behavior. Anomalies are then aggregated to events and further analyzed. When no evaluation data is supplied during
    training, the prediction and event analysis steps are skipped.
    Runs the entire fault detection module chain in one function call. Sections of this function call are:
    1. Data Loading and verification
    2. Config selection and optimization
    3. AnomalyDetector training
    4. AnomalyDetector prediction
    5. Event aggregation
    6. ARCANA-Analysis of detected events
    7. Visualization of output

    Args:
        csv_data_path (Optional[str]): Path to a csv-file containing tabular data which must contain training data for
            the autoencoder. When running in prediction mode with a pre-trained model, this can be ``None`` to skip
            loading training data. This data can also contain test data for evaluation, but in this case
            ``train_test_column`` and optionally ``train_test_mapping`` must be provided.
        csv_test_data_path (Optional str): Path to a csv file containing test data for evaluation. If test data is
            provided in both ways (i.e. via csv_test_data_path and in csv_data_path + train_test_column) then both test
            data sets will be fused into one. Default is None.
        train_test_column_name (Optional str): Name of the column which specifies which part of the data in
            csv_data_path is training data and which is test data. If this column does not contain boolean values or
            values which can be cast into boolean values, then train_test_mapping must be provided.
            True values are interpreted as training data. Default is None.
        train_test_mapping (Optional dict): Dictionary which defines a mapping of all non-boolean values in the
            train_test_column to booleans. Keys of the dictionary must be values from train_test_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. Default is None.
        time_column_name (Optional str): Name of the column containing time stamp information.
        status_data_column_name (Optional str): Name of the column which specifies the status of each row in
            csv_data_path. The status is used to define which rows represent normal behavior (i.e. which rows can be
            used for the autoencoder training) and which rows contain anomalous behavior. If this column does not
            contain boolean values, status_mapping must be provided. If status_data_column_name is not provided, all
            rows in csv_data_path are assumed to be normal and a warning will be logged. Default is None.
        status_mapping (Optional dict): Dictionary which defines a mapping of all non-boolean values in the
            status_data_column to booleans. Keys of the dictionary must be values from status_data_column, and they must
            have a datatype which can be cast to the datatype of train_test_column. Values of this dictionary must be
            booleans or at least castable to booleans. True values are interpreted as normal status. Default is None.
        status_label_confidence_percentage (Optional float): Specifies how sure the user is that the provided status 
            labels and derived normal_indexes are correct. This determines the quantile for quantile threshold method.
            Default is 0.95.
        features_to_exclude (Optional[List[str]]): List of column names which are present in the csv-files but which
            should be ignored for this failure detection run. Default is None.
        angle_features (Optional[List[str]]): List of column names which represent angle-features. This enables a
            specialized preprocessing of angle features, which might otherwise hinder the failure detection process.
            Default is None.
        automatic_optimization (bool): If True, an automatic hyperparameter optimization is done based on the dimension
            of the provided dataset. Default is True.
        enable_debug_plots (bool): If True advanced information for debugging is added to the result plots.
            Default is False.
        min_anomaly_length (int): Minimal number of consecutive anomalies (i.e. data points with an anomaly score above
            the FaultDetector threshold) to define an anomaly event when no explicit critical event length is provided.
        critical_event_min_length (Optional[int]): Override for the number of consecutive anomalous timestamps required
            to flag an event as critical. When ``None`` the value from the configuration or ``min_anomaly_length`` is
            used.
        critical_event_min_duration (Optional[Union[str, float, int]]): Minimum duration that anomalies must span to be
            considered critical. String values are parsed via :func:`pandas.to_timedelta` while numeric values are
            interpreted as seconds. When ``None`` the duration constraint is ignored.
        save_dir (Optional[str]): Directory to save the output plots. If not provided, the plots are not saved.
            Defaults to None.
        mode (Literal['train', 'predict']): Determines whether the detector should train a new model or load an
            existing one for prediction. Defaults to 'train'.
        model_path (Optional[str]): Path to a previously saved model directory. Required when mode='predict'.
        model_directory (Optional[str]): Directory where trained model artifacts should be stored. Defaults to the
            FaultDetector default when not provided.
        model_subdir (Optional[str]): Optional subdirectory inside ``model_directory`` that should be used when saving
            models. If not provided, a timestamp-based folder name is used.
        model_name (Optional[str]): Custom directory name for the saved model artifacts. Defaults to 'trained_model'.
        asset_name (Optional[str]): Identifier of the asset whose data is being analysed. When running in prediction
            mode this name is used to create a dedicated folder inside ``prediction_output``. If omitted the folder name
            is inferred from ``csv_test_data_path`` when possible.
        overwrite_models (bool): When True, allow existing model artefacts in ``model_directory`` to be overwritten
            during training. Defaults to False.
        rca_ignore_features (Optional[List[str]]): Additional feature names or wildcard patterns that should be ignored
            by the root cause analysis module during prediction. Values provided here extend the patterns that are stored
            in the persisted model configuration.

    Returns:
        Tuple(FaultDetectionResult, pd.DataFrame, List[Dict[str, Any]], Optional[ModelMetadata]): FaultDetectionResult
        object with the
            following DataFrames:

            - predicted_anomalies: DataFrame with a column 'anomaly' (bool).
            - reconstruction: DataFrame with reconstruction of the sensor data with timestamp as index.
            - deviations: DataFrame with reconstruction errors.
            - anomaly_score: DataFrame with anomaly scores for each timestamp.
            - bias_data: DataFrame with ARCANA results with timestamp as index. None if ARCANA was not run.
            - arcana_losses: DataFrame containing recorded values for all losses in ARCANA. None if ARCANA was not run.
            - tracked_bias: List of DataFrames. None if ARCANA was not run.

        and the detected anomaly events as dataframe. When run in prediction mode, all DataFrames contained in the
        ``FaultDetectionResult`` are additionally persisted as CSV files under ``prediction_output/<asset_name>``. The
        second return value contains event metadata while the third element of the tuple holds a list of dictionaries
        with detailed information for every detected event (sensor data slices and ARCANA summaries).

        When run in training mode, the tuple additionally contains ModelMetadata with information about the saved
        model. In prediction mode, the metadata element is None.
    """
    if mode not in {'train', 'predict'}:
        raise ValueError(f"Unsupported mode '{mode}'. Please choose 'train' or 'predict'.")

    if mode == 'predict' and model_path is None:
        raise ValueError('`model_path` must be provided when running in predict mode.')

    logger.info('Starting Automated Energy Fault Detection and Identification (mode=%s).', mode)
    logger.info('Loading Data...')
    if csv_data_path is None and csv_test_data_path is None:
        raise ValueError('At least one data source must be provided via `csv_data_path` or `csv_test_data_path`.')

    train_data, train_normal_index, test_data = load_train_test_data(csv_data_path=csv_data_path,
                                                                     csv_test_data_path=csv_test_data_path,
                                                                     train_test_column_name=train_test_column_name,
                                                                     train_test_mapping=train_test_mapping,
                                                                     time_column_name=time_column_name,
                                                                     status_data_column_name=status_data_column_name,
                                                                     status_mapping=status_mapping)
    model_metadata: Optional[ModelMetadata] = None

    if mode == 'train':
        logger.info('Selecting suitable config...')
        config = select_config(train_data=train_data, normal_index=train_normal_index,
                               status_label_confidence_percentage=status_label_confidence_percentage,
                               features_to_exclude=features_to_exclude, angles=angle_features,
                               automatic_optimization=automatic_optimization)
        logger.info('Training a Normal behavior model.')
        fault_detector_kwargs = {}
        if model_directory is not None:
            fault_detector_kwargs['model_directory'] = model_directory
        if model_subdir is not None:
            fault_detector_kwargs['model_subdir'] = model_subdir
        anomaly_detector = FaultDetector(config=config, **fault_detector_kwargs)
        model_metadata = anomaly_detector.fit(
            sensor_data=train_data,
            normal_index=train_normal_index,
            model_name=model_name,
            overwrite_models=overwrite_models,
        )
        if model_metadata.model_path:
            logger.info('Saved trained model to %s.', model_metadata.model_path)
        else:
            logger.info('Model was trained but not saved to disk.')
        root_cause_analysis = False
    else:
        logger.info('Loading pre-trained model from %s.', model_path)
        fallback_config_path = Path(__file__).resolve().parent.parent / 'base_config.yaml'
        fallback_config = Config(str(fallback_config_path))
        if rca_ignore_features:
            rca_config = fallback_config.config_dict.setdefault('root_cause_analysis', {})
            existing_patterns = rca_config.get('ignore_features', []) or []
            merged_patterns = list(dict.fromkeys([*existing_patterns, *rca_ignore_features]))
            rca_config['ignore_features'] = merged_patterns
        print("Fallback config:",fallback_config)
        anomaly_detector = FaultDetector(config=fallback_config)
        print("anomaly_detector:",anomaly_detector.config.arcana_params)
        anomaly_detector.load_models(model_path=model_path)
        print("anomaly_detector after load:",anomaly_detector.config.arcana_params)
        root_cause_analysis = True

    logger.info('Evaluating Test data based on the learned normal behavior.')
    has_test_data = test_data is not None and not test_data.empty

    if not has_test_data:
        if mode == 'predict':
            raise ValueError('Test data is required when running in predict mode.')
        logger.info('No test data provided; skipping prediction and event analysis for training run.')
        empty_results = _create_empty_prediction_result()
        return empty_results, pd.DataFrame(), [], model_metadata

    prediction_results = anomaly_detector.predict(sensor_data=test_data, root_cause_analysis=root_cause_analysis)
    predicted_anomalies = prediction_results.predicted_anomalies.copy()
    anomalies = predicted_anomalies['anomaly']

    config_min_length = getattr(anomaly_detector.config, 'critical_event_min_length', None) if anomaly_detector.config else None
    config_min_duration = getattr(anomaly_detector.config, 'critical_event_min_duration', None) if anomaly_detector.config else None

    effective_min_length: Optional[int]
    if critical_event_min_length is not None:
        effective_min_length = critical_event_min_length
    elif config_min_length is not None:
        effective_min_length = config_min_length
    else:
        effective_min_length = min_anomaly_length

    if effective_min_length is not None and effective_min_length < 1:
        effective_min_length = None

    effective_min_duration = (critical_event_min_duration
                              if critical_event_min_duration is not None
                              else config_min_duration)

    # Find anomaly events
    event_meta_data, event_data_list = create_events(
        sensor_data=test_data,
        boolean_information=anomalies,
        min_event_length=effective_min_length,
        min_event_duration=effective_min_duration,
    )
    event_ids = list(range(1, len(event_data_list) + 1))
    if not event_meta_data.empty:
        event_meta_data = event_meta_data.copy()
        event_meta_data.insert(0, 'event_id', event_ids)
        event_meta_data['critical_event'] = True

    predicted_anomalies['event_id'] = pd.Series(pd.NA, index=predicted_anomalies.index, dtype='Int64')
    predicted_anomalies['critical_event'] = False

    for event_id, event_data in zip(event_ids, event_data_list):
        event_index = event_data.index
        predicted_anomalies.loc[event_index, 'event_id'] = event_id
        predicted_anomalies.loc[event_index, 'critical_event'] = True

    prediction_results.predicted_anomalies = predicted_anomalies
    arcana_mean_importance_list: List[pd.Series] = []
    arcana_loss_list: List[Optional[pd.DataFrame]] = []
    for i in range(len(event_meta_data)):
        logger.info(f'Analyzing anomaly events ({i + 1} of {len(event_meta_data)}).')
        event_data = event_data_list[i]
        arcana_mean_importances, arcana_losses = analyze_event(
            anomaly_detector=anomaly_detector,
            event_data=event_data,
            track_losses=enable_debug_plots)
        arcana_mean_importance_list.append(arcana_mean_importances)
        if arcana_losses is not None and not arcana_losses.empty:
            arcana_loss_list.append(arcana_losses)
        else:
            arcana_loss_list.append(None)
    logger.info('Generating Output Graphics.')
    logger.info(output_info)
    generate_output_plots(anomaly_detector=anomaly_detector, train_data=train_data, normal_index=train_normal_index,
                          test_data=test_data, arcana_losses=arcana_loss_list,
                          arcana_mean_importances=arcana_mean_importance_list,
                          event_meta_data=event_meta_data, save_dir=save_dir)
    if mode == 'predict':
        _save_prediction_results(
            prediction_results=prediction_results,
            csv_test_data_path=csv_test_data_path,
            save_dir=save_dir,
            asset_name=asset_name,
        )

    event_analysis: List[Dict[str, Any]] = []
    for event_id, event_data, importances, losses in zip(event_ids, event_data_list,
                                                         arcana_mean_importance_list, arcana_loss_list):
        event_analysis.append({
            'event_id': event_id,
            'event_data': event_data,
            'arcana_mean_importances': importances,
            'arcana_losses': losses,
        })

    return prediction_results, event_meta_data, event_analysis, model_metadata


def analyze_event(anomaly_detector: FaultDetector, event_data: pd.DataFrame, track_losses: bool
                  ) -> Tuple[pd.Series, pd.DataFrame]:
    """ Runs root cause analysis for detected anomaly events.

    Args:
        anomaly_detector (FaultDetector): trained AnomalyDetector instance.
        event_data (pd.DataFrame): data from a detected anomaly event
        track_losses (bool): If True ARCANA-losses are tracked. 

    Returns:
        importances (pd.Series): Series of importance values for every feature in event_data.
        tracked_losses (pd.DataFrame): Potentially empty DataFrame containing recorded ARCANA losses.
    """
    bias, tracked_losses, _ = anomaly_detector.run_root_cause_analysis(sensor_data=event_data,
                                                                       track_losses=track_losses,
                                                                       track_bias=False)
    importances_mean = calculate_mean_arcana_importances(bias_data=bias)
    return importances_mean, tracked_losses


def _create_empty_prediction_result() -> FaultDetectionResult:
    """Build an empty FaultDetectionResult used when no evaluation data is provided."""

    predicted_anomalies = pd.DataFrame(
        columns=[
            'anomaly',
            'critical_event',
            'event_id',
            'behaviour',
            'anamoly_score',
            'threshold_score',
            'cumulative_anamoly_score',
            'anamolous fields',
        ]
    )

    return FaultDetectionResult(
        predicted_anomalies=predicted_anomalies,
        reconstruction=pd.DataFrame(),
        recon_error=pd.DataFrame(),
        anomaly_score=pd.DataFrame(columns=['value']),
        bias_data=None,
        arcana_losses=None,
        tracked_bias=None,
    )


def _save_prediction_results(prediction_results: FaultDetectionResult,
                             csv_test_data_path: Optional[str],
                             save_dir: Optional[str],
                             asset_name: Optional[str]) -> None:
    """Persist prediction artefacts to disk when running in predict mode."""

    base_directory: Path
    if save_dir is not None:
        base_directory = Path(save_dir)
    elif csv_test_data_path is not None:
        base_directory = Path(csv_test_data_path).resolve().parent
    else:
        base_directory = Path.cwd()

    inferred_asset_name: str
    if asset_name:
        inferred_asset_name = asset_name
    elif csv_test_data_path is not None:
        csv_path = Path(csv_test_data_path)
        inferred_asset_name = csv_path.resolve().parent.name or csv_path.stem
    else:
        inferred_asset_name = 'asset'

    output_directory = base_directory / 'prediction_output' / inferred_asset_name
    prediction_results.save(str(output_directory))
