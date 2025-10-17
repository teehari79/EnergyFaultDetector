"""Command line utility for launching bulk training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from energy_fault_detector.main import Options, load_options_from_yaml, run_bulk_training


DEFAULT_TRAINING_DIRECTORY = Path("~/datasets/bulk_train")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by the bulk training helper script."""

    parser = argparse.ArgumentParser(
        description=(
            "Train EnergyFaultDetector models for every 'train_*.csv' file in the "
            "supplied directory."
        )
    )
    parser.add_argument(
        "data_directory",
        nargs="?",
        default=str(DEFAULT_TRAINING_DIRECTORY),
        help=(
            "Directory containing CSV files named like 'train_<asset>.csv'. "
            "Defaults to '%(default)s'."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where trained models and optional artefacts should be stored.",
    )
    parser.add_argument(
        "--options",
        help=(
            "Path to a YAML file compatible with energy_fault_detector.main.Options. "
            "When omitted, the default configuration is used."
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


def validate_data_directory(path: str) -> Path:
    """Ensure ``path`` points to a directory containing ``train_*.csv`` files."""

    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"Data directory '{directory}' does not exist or is not a directory.")

    if not any(directory.glob("train_*.csv")):
        raise FileNotFoundError(
            f"No training files matching 'train_*.csv' were found inside '{directory}'."
        )

    return directory


def ensure_results_directory(path: str) -> Path:
    """Create ``path`` (including parents) if necessary and return it as :class:`Path`."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point used by ``if __name__ == '__main__'`` and tests."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        data_directory = validate_data_directory(args.data_directory)
        results_directory = ensure_results_directory(args.results_dir)
        options = load_options(args.options)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse exits

    trained_assets = run_bulk_training(
        training_directory=str(data_directory),
        options=options,
        results_dir=str(results_directory),
    )

    if trained_assets:
        print("Finished training the following assets:")
        for asset_name, model_path in trained_assets:
            if model_path:
                print(f"  - {asset_name}: {model_path}")
            else:
                print(f"  - {asset_name}: model path not reported")
    else:
        print("Bulk training finished but no assets were returned by run_bulk_training.")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
