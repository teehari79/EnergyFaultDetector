#!/usr/bin/env python
"""Utility script for running the quick fault detector in predict mode.

This executable mirrors the high level behaviour of :mod:`run.py`, but focuses on
loading a previously trained model and applying it to new data.  At a minimum the
user has to provide the path to the model directory and the CSV file that should be
evaluated.  Optionally a dedicated training dataset as well as metadata about the
input columns can be supplied.

Example usage::

    python run_predict.py \
        --model-path path/to/model_directory \
        --csv-predict-data path/to/predict.csv \
        --time-column time_stamp

The script prints a short textual summary of the run and leaves the detailed
artefacts (plots, CSVs) to the underlying ``quick_fault_detector`` implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from energy_fault_detector.quick_fault_detection import quick_fault_detector


DEFAULT_MODEL_PATH = (
    r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\models"
    r"\asset_0\20251007_154309"
)
DEFAULT_PREDICT_DATA_PATH = (
    r"D:\Personal\Ideas\Wind turbine\CARE_To_Compare\CARE_To_Compare\Wind Farm B\asset_files"
    r"\predict_5.csv"
)
DEFAULT_TIME_COLUMN = "time_stamp"


def _parse_mapping(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a JSON dictionary supplied via the command line.

    Args:
        value: Raw argument string.

    Returns:
        Parsed dictionary or ``None`` when ``value`` is ``None``.

    Raises:
        argparse.ArgumentTypeError: If the value is not valid JSON or does not
            decode to a dictionary.
    """

    if value is None:
        return None

    try:
        mapping = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - handled at runtime
        raise argparse.ArgumentTypeError(
            f"Failed to decode mapping '{value}': {exc}"
        ) from exc

    if not isinstance(mapping, dict):
        raise argparse.ArgumentTypeError("Mapping must decode to a JSON object (dict).")

    return mapping


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser used by the script."""

    parser = argparse.ArgumentParser(
        description="Run the quick fault detector in predict mode using a saved model.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the directory that contains the previously trained model. "
            "Defaults to the predefined wind farm model when not supplied."
        ),
    )
    parser.add_argument(
        "--csv-predict-data",
        default=DEFAULT_PREDICT_DATA_PATH,
        help=(
            "CSV file that contains the data points to analyse for anomalies. "
            "Defaults to the provided wind farm dataset when not supplied."
        ),
    )
    parser.add_argument(
        "--csv-train-data",
        help=(
            "Optional CSV file that contains the training data. When omitted, no "
            "additional training is performed and the provided model is reused."
        ),
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=(
            "Name of the timestamp column in the provided CSV files. Defaults to "
            "'time_stamp' for the predefined dataset."
        ),
    )
    parser.add_argument(
        "--status-column",
        help=(
            "Name of the column describing the operating status of the asset."
            " Used to identify rows that represent normal behaviour."
        ),
    )
    parser.add_argument(
        "--status-mapping",
        type=_parse_mapping,
        help="JSON dictionary mapping status values to booleans (e.g. '{\"OK\": true}').",
    )
    parser.add_argument(
        "--train-test-column",
        help=(
            "Column indicating whether a row belongs to the training (True) or"
            " prediction (False) portion of the dataset."
        ),
    )
    parser.add_argument(
        "--train-test-mapping",
        type=_parse_mapping,
        help="JSON dictionary mapping train/test column values to booleans.",
    )
    parser.add_argument(
        "--min-anomaly-length",
        type=int,
        default=18,
        help="Minimal number of consecutive anomalies to form an event (default: 18).",
    )
    parser.add_argument(
        "--save-dir",
        help="Directory where diagnostic output (plots, CSVs) should be written.",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        help="Enable creation of additional debug plots during prediction.",
    )
    return parser


def validate_paths(model_path: str, csv_predict_data: str, csv_train_data: Optional[str]) -> None:
    """Ensure that the provided filesystem paths point to existing resources."""

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    predict_path = Path(csv_predict_data)
    if not predict_path.is_file():
        raise FileNotFoundError(f"Prediction data file '{predict_path}' was not found.")

    if csv_train_data is not None:
        train_path = Path(csv_train_data)
        if not train_path.is_file():
            raise FileNotFoundError(f"Training data file '{train_path}' was not found.")


def main() -> None:
    """Entry point for command line execution."""

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_paths(
        model_path=args.model_path,
        csv_predict_data=args.csv_predict_data,
        csv_train_data=args.csv_train_data,
    )

    train_data_path = args.csv_train_data

    prediction_results, event_metadata, _ = quick_fault_detector(
        csv_data_path=train_data_path,
        csv_test_data_path=args.csv_predict_data,
        train_test_column_name=args.train_test_column,
        train_test_mapping=args.train_test_mapping,
        time_column_name=args.time_column,
        status_data_column_name=args.status_column,
        status_mapping=args.status_mapping,
        min_anomaly_length=args.min_anomaly_length,
        save_dir=args.save_dir,
        enable_debug_plots=args.debug_plots,
        mode="predict",
        model_path=args.model_path,
    )

    print("Prediction finished successfully.")
    print(f"Number of detected events: {len(event_metadata)}")
    print("Anomaly score preview:")
    print(prediction_results.anomaly_score.head())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

