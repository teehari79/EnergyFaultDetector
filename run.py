"""Command line helper for training quick fault detector models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from energy_fault_detector.main import Options, load_options_from_yaml
from energy_fault_detector.quick_fault_detection import quick_fault_detector


DEFAULT_MODEL_NAME = "trained_model"
# Paths below can be customised so the helper script runs without supplying
# command line arguments. They intentionally expand user home directories to
# make it easy to point at data stored outside of the repository.
DEFAULT_TRAINING_DATA_PATH = Path("~/datasets/train.csv")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by the training helper script."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the quick fault detector on a single CSV dataset."
            " Optional evaluation data can be supplied via the options YAML"
            " file to export anomaly predictions and detected events."
        )
    )
    parser.add_argument(
        "csv_data_path",
        nargs="?",
        default=str(DEFAULT_TRAINING_DATA_PATH),
        help=(
            "Path to the CSV file that should be used for training. Defaults to "
            "'%(default)s'."
        ),
    )
    parser.add_argument(
        "--options",
        help=(
            "Path to a YAML file compatible with energy_fault_detector.main.Options. "
            "The file can specify evaluation data, column mappings, and other"
            " advanced settings."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where prediction artefacts should be stored.",
    )
    parser.add_argument(
        "--model-dir",
        help=(
            "Directory where trained model artefacts should be saved."
            " Defaults to the results directory when omitted."
        ),
    )
    parser.add_argument(
        "--model-subdir",
        help="Optional subdirectory inside the model directory used for saving runs.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Optional folder name for the saved model artefacts.",
    )
    parser.add_argument(
        "--asset-name",
        help=(
            "Identifier for the analysed asset. When provided, prediction outputs"
            " are written to a dedicated 'prediction_output/<asset-name>' folder."
        ),
    )
    return parser


def load_options(options_path: Optional[str]) -> Options:
    """Load option overrides from ``options_path`` if provided."""

    if options_path is None:
        return Options()

    path = Path(options_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Options file '{path}' does not exist.")

    return load_options_from_yaml(str(path))


def ensure_file_exists(path: str) -> Path:
    """Return ``path`` as :class:`Path` and ensure it points to a file."""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Training data file '{file_path}' does not exist.")
    return file_path


def ensure_directory(path: Optional[str], fallback: Path) -> Path:
    """Create ``path`` (or ``fallback``) if necessary and return it."""

    if path is None:
        directory = fallback
    else:
        directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point used by ``if __name__ == '__main__'`` and tests."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        training_csv = ensure_file_exists(args.csv_data_path)
        options = load_options(args.options)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse exits

    results_directory = ensure_directory(args.results_dir, Path("results").expanduser())
    model_directory = ensure_directory(args.model_dir, results_directory)

    csv_test_data_path = options.csv_test_data_path
    if csv_test_data_path is not None:
        csv_test_data_path = str(Path(csv_test_data_path).expanduser())

    prediction_results, event_metadata, _event_details, model_metadata = quick_fault_detector(
        csv_data_path=str(training_csv),
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
        save_dir=str(results_directory),
        mode="train",
        model_directory=str(model_directory),
        model_subdir=args.model_subdir,
        model_name=args.model_name,
        asset_name=args.asset_name,
    )

    if prediction_results.predicted_anomalies.empty:
        print("Training finished. No evaluation data was provided, so no prediction artefacts were saved.")
    else:
        prediction_results.save(str(results_directory))
        print(f"Prediction artefacts written to: {results_directory}")

        if not event_metadata.empty:
            events_path = results_directory / "events.csv"
            event_metadata.to_csv(events_path, index=False)
            print(f"Event metadata stored at: {events_path}")

    if model_metadata is not None and model_metadata.model_path:
        print(f"Model saved to: {model_metadata.model_path}")
    else:
        print("Model training completed but no model path was reported.")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
