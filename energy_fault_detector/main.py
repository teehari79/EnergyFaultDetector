"""Quick energy fault detector CLI tool, to try out the EnergyFaultDetector model on a specific dataset."""

import os
import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

logger = logging.getLogger('energy_fault_detector')
here = os.path.abspath(os.path.dirname(__file__))


@dataclass
class Options:
    csv_test_data_path: Optional[str] = None
    train_test_column_name: Optional[str] = None
    train_test_mapping: Optional[dict] = None
    time_column_name: Optional[str] = None
    status_data_column_name: Optional[str] = None
    status_mapping: Optional[dict] = None
    status_label_confidence_percentage: float = 0.95
    min_anomaly_length: int = 18
    critical_event_min_length: Optional[int] = None
    critical_event_min_duration: Optional[str] = None
    features_to_exclude: List[str] = field(default_factory=list)
    angle_features: List[str] = field(default_factory=list)
    automatic_optimization: bool = True
    enable_debug_plots: bool = False


def load_options_from_yaml(file_path: str) -> Options:
    """Load options from a YAML file and return an Options dataclass."""
    with open(file_path, 'r') as file:
        options_dict = yaml.safe_load(file)
        return Options(**options_dict)


def main():
    parser = argparse.ArgumentParser(
        description='''
        Quick Fault Detection Tool for Energy Systems. This tool analyzes provided data using an
        autoencoder-based approach to identify anomalies based on learned normal behavior.
        Anomalies are then aggregated into events for further analysis.

        Required Arguments:
        - csv_data_path: Path to a CSV file containing training data.

        Optional Arguments (via YAML file):
        - options: Path to a YAML file containing additional options.

        Example YAML file structure:
            csv_test_data_path: "path/to/test_data.csv"
            train_test_column_name: "train_test"      # true = training data
            train_test_mapping:
                train: true
                test: false
            time_column_name: "timestamp"
            status_data_column_name: "status"         # true = normal behaviour
            status_mapping: 
                production: true
                service: false
                error: false
            status_label_confidence_percentage: 0.95  # contamination level
            min_anomaly_length: 18
            features_to_exclude: 
              - do_not_use_this_feature_1
              - do_not_use_this_feature_2
            angle_features:
              - angle1
              - angle2
            automatic_optimization: true
            enable_debug_plots: false
        ''',

        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'csv_data_path',
        type=str,
        help='Path to a CSV file containing training data.'
    )
    parser.add_argument(
        '--options',
        type=str,
        help='Path to a YAML file containing additional options.',
        default=None,
        required=False,
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Path to a directory where results will be saved.',
        default='results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'bulk_train'],
        default='train',
        help=(
            "Run in 'train' mode to fit a new model, 'predict' to load an existing one, "
            "or 'bulk_train' to train models for every 'train_*.csv' file in a directory."
        )
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to a previously saved model directory (required for predict mode).',
        default=None
    )
    parser.add_argument(
        '--c2c_example',
        action='store_true',
        help='Whether to use default settings for a CARE2Compare dataset.',
    )

    args = parser.parse_args()
    logger.info(f"CSV Data Path: {args.csv_data_path}")

    if args.c2c_example:
        logger.info("Using default settings for CARE2Compare dataset.")
    else:
        logger.info(f"Options YAML: {args.options}")

    logger.info(f"Results Directory: {args.results_dir}")
    logger.info(f"Execution Mode: {args.mode}")
    os.makedirs(args.results_dir, exist_ok=True)

    options = Options()  # Initialize with default values
    if args.options:
        options = load_options_from_yaml(args.options)
    elif args.c2c_example:
        options = load_options_from_yaml(os.path.join(here, 'c2c_options.yaml'))

    print(options)

    if args.mode == 'predict' and args.model_path is None:
        parser.error('--model_path must be provided when --mode is set to predict.')

    # Call the quick_fault_detector function with parsed arguments
    try:
        from .quick_fault_detection import quick_fault_detector
        if args.mode == 'bulk_train':
            run_bulk_training(
                training_directory=args.csv_data_path,
                options=options,
                results_dir=args.results_dir,
            )
            logger.info('Bulk training completed. Models saved under %s.', args.results_dir)
            return

        prediction_results, event_meta_data, _event_details, model_metadata = quick_fault_detector(
            csv_data_path=args.csv_data_path,
            csv_test_data_path=options.csv_test_data_path,
            train_test_column_name=options.train_test_column_name,
            train_test_mapping=options.train_test_mapping,
            time_column_name=options.time_column_name,
            status_data_column_name=options.status_data_column_name,
            status_mapping=options.status_mapping,
            status_label_confidence_percentage=options.status_label_confidence_percentage,
            features_to_exclude=options.features_to_exclude,
            angle_features=options.angle_features,
            automatic_optimization=options.automatic_optimization,
            enable_debug_plots=options.enable_debug_plots,
            min_anomaly_length=options.min_anomaly_length,
            critical_event_min_length=options.critical_event_min_length,
            critical_event_min_duration=options.critical_event_min_duration,
            save_dir=args.results_dir,
            mode=args.mode,
            model_path=args.model_path,
        )

        if prediction_results.predicted_anomalies.empty:
            logger.info('No test data provided; skipping prediction artefact export.')
        else:
            logger.info(f'Fault detection completed. Results are saved in {args.results_dir}.')
            prediction_results.save(args.results_dir)
            predicted_anomalies_path = os.path.join(args.results_dir, 'predicted_anomalies.csv')
            logger.info('Prediction data stored at %s.', predicted_anomalies_path)
            if not event_meta_data.empty:
                events_path = os.path.join(args.results_dir, 'events.csv')
                event_meta_data.to_csv(events_path, index=False)
                logger.info('Event metadata stored at %s.', events_path)

            anomaly_count = int(prediction_results.predicted_anomalies['anomaly'].sum())
            logger.info('Detected %d anomalous timestamps grouped into %d events.', anomaly_count,
                        len(event_meta_data))

        if model_metadata is not None and model_metadata.model_path:
            logger.info('Trained model saved to %s.', model_metadata.model_path)
            print(f"Model saved to: {model_metadata.model_path}")
        elif args.mode == 'predict' and args.model_path is not None:
            logger.info('Predictions generated using model at %s.', args.model_path)

    except Exception as e:
        logger.error(f'An error occurred: {e}')


EXISTING_MODEL_REQUIRED_ENTRIES = (
    "data_preprocessor",
    "autoencoder",
    "threshold_selector",
    "anomaly_score",
    "config.yaml",
)


def _find_existing_model_path(asset_results_dir: Path) -> Optional[Path]:
    """Return the most recent model directory for ``asset_results_dir`` when artefacts exist."""

    models_dir = asset_results_dir / "models"
    if not models_dir.is_dir():
        return None

    candidate_dirs = [path for path in models_dir.iterdir() if path.is_dir()]
    if not candidate_dirs:
        return None

    # Sort newest first to return the most recent valid model path.
    candidate_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for candidate in candidate_dirs:
        expected_paths = [candidate / name for name in EXISTING_MODEL_REQUIRED_ENTRIES]
        if all(path.exists() for path in expected_paths):
            return candidate

    return None


def run_bulk_training(
    training_directory: str,
    options: Options,
    results_dir: str,
    existing_model_behavior: str = "skip",
    run_prediction: bool = True,
) -> List[Tuple[str, str]]:
    """Train models for every ``train_*.csv`` file within ``training_directory``.

    Args:
        training_directory: Directory containing the training CSV files.
        options: Configuration options used for training and prediction.
        results_dir: Directory where artefacts generated during training are stored.
        existing_model_behavior: Whether to skip or overwrite existing trained models.
        run_prediction: When ``True`` attempt to load matching ``predict_*.csv`` files
            and generate evaluation artefacts for each asset. When ``False`` the
            training run skips the prediction and evaluation phase entirely.
    """

    if existing_model_behavior not in {"skip", "overwrite"}:
        raise ValueError(
            "existing_model_behavior must be either 'skip' or 'overwrite', "
            f"received {existing_model_behavior!r}."
        )

    from .quick_fault_detection import quick_fault_detector

    data_dir = Path(training_directory).expanduser()
    if not data_dir.is_dir():
        raise ValueError(f'Bulk training expected a directory but received {training_directory!r}.')

    train_files = sorted(data_dir.glob('train_*.csv'))
    if not train_files:
        raise ValueError(f'No training files starting with "train_" found in {training_directory!r}.')

    os.makedirs(results_dir, exist_ok=True)
    trained_assets: List[Tuple[str, str]] = []

    for train_file in train_files:
        print("Train file names:",train_file.name,re.search(r'train_(\d+)\.csv$', train_file.name))
        # match = re.search(r'train_(\d+)\\.csv$', train_file.name)
        match = re.search(r'train_(\d+)\.csv$', train_file.name)
        if not match:
            logger.warning('Skipping %s because no asset number could be derived from the filename.', train_file)
            continue

        asset_number = match.group(1)
        asset_name = f'asset_{asset_number}'
        asset_results_dir = Path(results_dir) / asset_name
        asset_results_dir.mkdir(parents=True, exist_ok=True)

        test_file = train_file.with_name(f'predict_{asset_number}.csv')
        csv_test_data_path = None
        if run_prediction and test_file.exists():
            csv_test_data_path = str(test_file)
        elif not run_prediction:
            logger.info('Skipping prediction for %s because bulk prediction is disabled.', asset_name)
        else:
            logger.info('No prediction file found for %s; skipping evaluation.', asset_name)

        existing_model_path = _find_existing_model_path(asset_results_dir)
        if existing_model_path is not None and existing_model_behavior == "skip":
            logger.info(
                "Skipping %s because existing model artefacts were found at %s.",
                asset_name,
                existing_model_path,
            )
            trained_assets.append((asset_name, str(existing_model_path)))
            continue

        overwrite_models = existing_model_behavior == "overwrite"

        prediction_results, event_meta_data, _details, model_metadata = quick_fault_detector(
            csv_data_path=str(train_file),
            csv_test_data_path=csv_test_data_path,
            train_test_column_name=options.train_test_column_name,
            train_test_mapping=options.train_test_mapping,
            time_column_name=options.time_column_name,
            status_data_column_name=options.status_data_column_name,
            status_mapping=options.status_mapping,
            status_label_confidence_percentage=options.status_label_confidence_percentage,
            features_to_exclude=options.features_to_exclude,
            angle_features=options.angle_features,
            automatic_optimization=options.automatic_optimization,
            enable_debug_plots=options.enable_debug_plots,
            min_anomaly_length=options.min_anomaly_length,
            critical_event_min_length=options.critical_event_min_length,
            critical_event_min_duration=options.critical_event_min_duration,
            save_dir=str(asset_results_dir),
            mode='train',
            model_directory=str(asset_results_dir),
            model_subdir='models',
            model_name='trained_model',
            asset_name=asset_name,
            overwrite_models=overwrite_models,
        )

        if (
            run_prediction
            and csv_test_data_path
            and prediction_results is not None
            and not prediction_results.predicted_anomalies.empty
        ):
            prediction_results.save(str(asset_results_dir))
            if not event_meta_data.empty:
                events_path = asset_results_dir / 'events.csv'
                event_meta_data.to_csv(events_path, index=False)

        if model_metadata is not None and model_metadata.model_path:
            logger.info('Model for %s saved to %s.', asset_name, model_metadata.model_path)
            trained_assets.append((asset_name, model_metadata.model_path))
        else:
            trained_assets.append((asset_name, ''))

    return trained_assets


if __name__ == '__main__':
    main()
