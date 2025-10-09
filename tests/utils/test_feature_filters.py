import pytest

pd = pytest.importorskip('pandas')

import pandas as pd

from energy_fault_detector.utils.feature_filters import (
    mask_ignored_features,
    resolve_ignored_columns,
)


def test_resolve_ignored_columns_matches_and_reports_unmatched():
    columns = ['windspeed_avg', 'output_power', 'temperature']
    patterns = ['windspeed*', 'non_existent']

    ignored, unmatched = resolve_ignored_columns(columns, patterns)

    assert ignored == {'windspeed_avg'}
    assert unmatched == {'non_existent'}


def test_resolve_ignored_columns_matches_case_insensitively():
    columns = ['Power_58_Avg', 'Other']
    patterns = ['power_58_avg']

    ignored, unmatched = resolve_ignored_columns(columns, patterns)

    assert ignored == {'Power_58_Avg'}
    assert unmatched == set()


def test_mask_ignored_features_sets_columns_to_zero():
    df = pd.DataFrame({
        'windspeed_avg': [1.0, -2.0],
        'temperature': [3.0, 4.0],
    })

    masked, ignored, unmatched = mask_ignored_features(df, ['windspeed*'])

    assert ignored == {'windspeed_avg'}
    assert unmatched == set()
    assert masked['windspeed_avg'].tolist() == [0.0, 0.0]
    assert masked['temperature'].tolist() == [3.0, 4.0]
    # original DataFrame is not modified
    assert df['windspeed_avg'].tolist() == [1.0, -2.0]

