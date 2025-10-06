from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from energy_fault_detector.utils.asset_dataset_splitter import split_asset_datasets


def _write_csv(path: Path, data: pd.DataFrame) -> None:
    path.write_text(data.to_csv(sep=";", index=False))


def test_split_asset_datasets(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    data = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a", "asset_a", "asset_b"],
            "train_test": ["train", "prediction", "train", "prediction"],
            "train_test_bool": [True, False, True, False],
            "status_type_id": [0, 1, 3, 2],
            "status_type_bool": [False, True, True, False],
            "temperature_avg": [10, 11, 12, 13],
            "temperature_max": [20, 21, 22, 23],
            "sensor_value": [100, 110, 120, 130],
            "time_stamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )

    _write_csv(input_dir / "dataset.csv", data)

    split_asset_datasets(input_dir)

    train_a = pd.read_csv(input_dir / "train_asset_a.csv", sep=";")
    predict_a = pd.read_csv(input_dir / "predict_asset_a.csv", sep=";")
    predict_b = pd.read_csv(input_dir / "predict_asset_b.csv", sep=";")

    assert set(train_a.columns) == {"temperature_avg", "sensor_value", "time_stamp"}
    assert set(predict_a.columns) == {"temperature_avg", "sensor_value", "time_stamp"}

    assert len(train_a) == 3  # includes both train rows and normal status rows
    assert len(predict_a) == 2  # prediction row + anomaly status row

    # Asset B only has a prediction row with normal status. It should appear in prediction output
    assert set(predict_b.columns) == {"temperature_avg", "sensor_value", "time_stamp"}
    assert len(predict_b) == 1
