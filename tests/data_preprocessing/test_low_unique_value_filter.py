
import unittest

import pandas as pd

from energy_fault_detector.data_preprocessing.low_unique_value_filter import LowUniqueValueFilter


class TestLowUniqueValueFilter(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.df = pd.DataFrame({
            'A': [1, 2, 2, 3, 4],  # Unique values: 4
            'B': [0, 0, 0, 0, 0],  # Unique values: 1 (to be dropped based on zero fraction)
            'C': [1, 1, 1, 1, 1],  # Unique values: 1 (to be dropped)
            'D': [1, 0, 1, 0, 1]  # Unique values: 2 (to be kept)
        })
        self.filter = LowUniqueValueFilter(min_unique_value_count=2, max_col_zero_frac=0.9)

    def test_fit_keeps_valid_columns(self):
        """Test that valid columns are kept after fitting."""
        self.filter.fit(self.df)
        expected_columns = ['A', 'D']
        self.assertListEqual(self.filter.feature_names_out_, expected_columns)

    def test_fit_drops_low_unique_columns(self):
        """Test that columns with low unique values are dropped."""
        self.filter.fit(self.df)
        expected_dropped_columns = ['B', 'C']
        self.assertListEqual(self.filter.columns_dropped_, expected_dropped_columns)

    def test_transform(self):
        """Test the transform method returns only the selected features."""
        self.filter.fit(self.df)
        transformed_df = self.filter.transform(self.df)
        expected_df = self.df[['A', 'D']]
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_inverse_transform(self):
        """Test the inverse transform method does nothing."""
        x_transformed = self.filter.inverse_transform(self.df)
        pd.testing.assert_frame_equal(x_transformed, self.df)

    def test_zero_fraction_threshold(self):
        """Test that a column is dropped when the zero fraction exceeds the threshold."""
        df_test = pd.DataFrame({
            'A': [0, 0, 0, 1, 1],  # Unique values: 2, Zero fraction: 0.6
            'B': [0, 0, 0, 0, 0],  # Unique values: 1, Zero fraction: 1.0 (to be dropped)
            'C': [1, 1, 1, 1, 1],  # Unique values: 1, Zero fraction: 0 (to be dropped)
            'D': [2, 2, 2, 3, 3]  # Unique values: 3, Zero fraction: 0
        })
        filter_test = LowUniqueValueFilter(min_unique_value_count=2, max_col_zero_frac=0.9)
        filter_test.fit(df_test)
        expected_dropped_columns = ['B', 'C']
        self.assertListEqual(filter_test.columns_dropped_, expected_dropped_columns)
