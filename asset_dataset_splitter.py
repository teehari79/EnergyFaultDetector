"""Utility for splitting asset datasets into train and prediction CSV files.

The module provides a command line interface that accepts the path to a
directory containing ``.csv`` files separated by semicolons. Each file is
expected to contain records for one or more assets identified by the
``asset_id`` column.

The script creates two output files per asset:

``train_<asset>.csv``
    Contains all records marked as ``train`` in the ``train_test`` column and
    all records whose ``status_type_id`` is either 0 or 2.

``predict_<asset>.csv``
    Contains all records whose ``status_type_id`` is 1, 3, 4 or 5 and all
    records whose ``train_test`` value is ``prediction`` regardless of the
    ``status_type_id`` value.

Before saving, the script removes helper columns as well as columns that
contain ``_max``, ``_min`` or ``_std`` from the resulting files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, MutableMapping

import pandas as pd

import csv


STATUS_TYPE_NORMAL = {"0", "2"}
STATUS_TYPE_ANOMALY = {"1", "3", "4", "5"}
DROP_COLUMNS = {
    "asset_id",
    "train_test",
    "train_test_bool",
    "status_type_id",
    "status_type_bool",
}
DROP_COLUMN_SUBSTRINGS = ("_max", "_min", "_std")


def _detect_delimiter(csv_file: Path) -> str:
    """Return the detected delimiter for ``csv_file`` with ``;`` as fallback."""

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return ";"


def _iter_asset_frames(
    csv_file: Path, *, chunksize: int | None = 100_000
) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield (asset_id, dataframe) pairs from ``csv_file`` without loading everything into memory."""

    delimiter = _detect_delimiter(csv_file)
    reader = pd.read_csv(csv_file, sep=delimiter, dtype=str, chunksize=chunksize)

    # ``pd.read_csv`` returns ``DataFrame`` when ``chunksize`` is ``None``.
    if isinstance(reader, pd.DataFrame):
        reader = [reader]

    for chunk in reader:
        if chunk.empty:
            continue

        if "asset_id" not in chunk.columns:
            raise ValueError(
                f"File '{csv_file}' does not contain required column 'asset_id'."
            )

        for asset_id, group in chunk.groupby("asset_id", sort=False):
            yield str(asset_id), group.reset_index(drop=True)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove helper columns and columns containing ``_max``/``_min``/``_std``."""

    to_drop = [col for col in df.columns if col in DROP_COLUMNS]
    to_drop.extend(
        col for col in df.columns if any(substr in col for substr in DROP_COLUMN_SUBSTRINGS)
    )
    return df.drop(columns=to_drop, errors="ignore")


def _get_lowercase_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df.get(column)
    if series is None:
        return pd.Series("", index=df.index, dtype=object)
    return series.fillna("").astype(str).str.lower()


def _get_status_series(df: pd.DataFrame) -> pd.Series:
    series = df.get("status_type_id")
    if series is None:
        return pd.Series("", index=df.index, dtype=object)
    return series.fillna("").astype(str)


def _build_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows belonging to the training split."""

    train_series = _get_lowercase_series(df, "train_test")
    status_series = _get_status_series(df)

    is_train = train_series == "train"
    is_normal_status = status_series.isin(STATUS_TYPE_NORMAL)
    return df[is_train | is_normal_status].copy()


def _build_predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows belonging to the prediction split."""

    train_series = _get_lowercase_series(df, "train_test")
    status_series = _get_status_series(df)

    is_prediction = train_series == "prediction"
    is_anomaly_status = status_series.isin(STATUS_TYPE_ANOMALY)
    return df[is_prediction | is_anomaly_status].copy()


def _append_asset_frames(
    asset_id: str,
    df: pd.DataFrame,
    output_dir: Path,
    header_written: MutableMapping[Path, bool],
) -> None:
    """Append data for ``asset_id`` to the corresponding train/predict CSV files."""

    if df.empty:
        return

    train_df = _clean_columns(_build_train_frame(df))
    predict_df = _clean_columns(_build_predict_frame(df))

    if not train_df.empty:
        train_path = output_dir / f"train_{asset_id}.csv"
        train_df.to_csv(
            train_path,
            sep=";",
            index=False,
            mode="a",
            header=not header_written.get(train_path, False),
        )
        header_written[train_path] = True

    if not predict_df.empty:
        predict_path = output_dir / f"predict_{asset_id}.csv"
        predict_df.to_csv(
            predict_path,
            sep=";",
            index=False,
            mode="a",
            header=not header_written.get(predict_path, False),
        )
        header_written[predict_path] = True


def split_asset_datasets(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    chunksize: int | None = 100_000,
) -> None:
    """Split datasets per asset into train and prediction CSV files.

    Args:
        input_dir: Directory containing the source ``.csv`` files.
        output_dir: Optional directory to store the results. When ``None`` the
            input directory is used.
        chunksize: Maximum number of rows per chunk read from the source files.
            ``None`` loads the entire file at once.
    """

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory '{input_dir}' does not exist or is not a directory")

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously generated files to avoid appending to stale data.
    for existing_file in output_dir.glob("train_*.csv"):
        existing_file.unlink()
    for existing_file in output_dir.glob("predict_*.csv"):
        existing_file.unlink()

    header_written: Dict[Path, bool] = {}

    for csv_file in sorted(input_dir.glob("*.csv")):
        for asset_id, asset_df in _iter_asset_frames(csv_file, chunksize=chunksize):
            _append_asset_frames(asset_id, asset_df, output_dir, header_written)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split asset datasets into train and prediction files")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the directory containing the source CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the input directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    split_asset_datasets(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
