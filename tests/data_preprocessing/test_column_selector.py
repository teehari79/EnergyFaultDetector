
from unittest import TestCase

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.utils.validation import NotFittedError

from energy_fault_detector.data_preprocessing.column_selector import ColumnSelector


class TestColumnSelector(TestCase):
    def setUp(self) -> None:
        self.column_selector = ColumnSelector(max_nan_frac_per_col=0.2, features_to_exclude=['sensor_6'])

        # generate data
        length = 10  # choose an even number for simplicity
        time_index = pd.date_range(start='1/1/2021', end='10/1/2021', periods=length)
        data = {'Sensor_1': list(range(length)),
                'Sensor_2': [np.nan] + list(range(1, length)),
                'Sensor_3': list(range(int(length / 2))) + [np.nan] * int(length / 2),
                'Sensor_4': [0.] + [np.nan] * (length - 1),
                'Sensor_5': [0.] * length,
                'SENsor_6': ['x'] * length}

        # input
        self.raw_dataframe = pd.DataFrame(index=time_index, data=data)

    def test_fit(self):
        # expected fit result
        expected_attributes = ["Sensor_1", "Sensor_2", 'Sensor_5']

        self.column_selector.fit(self.raw_dataframe)
        attributes = self.column_selector.feature_names_out_

        assert_array_equal(expected_attributes, attributes)

    def test_transform(self):
        # expected result
        expected_df = self.raw_dataframe[["Sensor_1", "Sensor_2", 'Sensor_5']]

        df = self.column_selector.fit_transform(self.raw_dataframe)
        assert_array_equal(expected_df.values, df.values)

    def test_missing_columns(self):
        missing_columns_dataframe = self.raw_dataframe[["Sensor_1"]]
        self.column_selector.fit(self.raw_dataframe)
        with self.assertRaises(ValueError):
            self.column_selector.transform(missing_columns_dataframe)

    def test_not_fitted(self):
        with self.assertRaises(NotFittedError):
            self.column_selector.transform(self.raw_dataframe)
