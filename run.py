"""Utility script to trigger bulk training without command-line arguments."""

from pathlib import Path

from energy_fault_detector.main import Options, load_options_from_yaml, run_bulk_training

# Update these paths before running the script.
DATA_DIRECTORY = Path("/path/to/training_folder")
RESULTS_DIRECTORY = Path("results")
OPTIONS_FILE = None  # Optionally point to a YAML configuration file.


def main() -> None:
    """Execute bulk training for all ``train_*.csv`` files in ``DATA_DIRECTORY``."""
    if OPTIONS_FILE is not None:
        options = load_options_from_yaml(str(OPTIONS_FILE))
    else:
        options = Options()

    run_bulk_training(
        training_directory=str(DATA_DIRECTORY),
        options=options,
        results_dir=str(RESULTS_DIRECTORY),
    )


if __name__ == "__main__":
    main()
