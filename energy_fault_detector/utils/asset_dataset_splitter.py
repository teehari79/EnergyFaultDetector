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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


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


def _read_asset_frames(path: Path) -> Dict[str, List[pd.DataFrame]]:
    """Read ``.csv`` files from ``path`` and group rows by ``asset_id``.

    Args:
        path: Directory containing the source ``.csv`` files.

    Returns:
        Mapping of asset identifier to a list of data frames containing rows
        for that asset.
    """

    asset_frames: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for csv_file in sorted(path.glob("*.csv")):
        df = pd.read_csv(csv_file, sep=";", dtype=str)
        if "asset_id" not in df.columns:
            raise ValueError(
                f"File '{csv_file}' does not contain required column 'asset_id'."
            )

        for asset_id, group in df.groupby("asset_id", sort=False):
            asset_frames[str(asset_id)].append(group.reset_index(drop=True))

    return asset_frames


def _combine_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine a list of frames into a single data frame."""

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def _save_asset_frames(asset_id: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Create train and prediction files for ``asset_id`` from ``df``."""

    if df.empty:
        return

    train_df = _clean_columns(_build_train_frame(df))
    predict_df = _clean_columns(_build_predict_frame(df))

    if not train_df.empty:
        train_path = output_dir / f"train_{asset_id}.csv"
        train_df.to_csv(train_path, sep=";", index=False)

    if not predict_df.empty:
        predict_path = output_dir / f"predict_{asset_id}.csv"
        predict_df.to_csv(predict_path, sep=";", index=False)


def split_asset_datasets(input_dir: Path, output_dir: Path | None = None) -> None:
    """Split datasets per asset into train and prediction CSV files.

    Args:
        input_dir: Directory containing the source ``.csv`` files.
        output_dir: Optional directory to store the results. When ``None`` the
            input directory is used.
    """

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory '{input_dir}' does not exist or is not a directory")

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_frames = _read_asset_frames(input_dir)
    for asset_id, frames in asset_frames.items():
        asset_df = _combine_frames(frames)
        _save_asset_frames(asset_id, asset_df, output_dir)


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
