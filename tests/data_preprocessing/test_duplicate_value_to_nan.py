
import unittest
import pandas as pd
import numpy as np

from energy_fault_detector.data_preprocessing.duplicate_value_to_nan import DuplicateValuesToNan


class DuplicateValuesToNanTests(unittest.TestCase):
    def setUp(self):
        data = {
            'A': [1, 2, 2, 3, 4, 5, 5],
            'B': [0, 0, 0, 0, 0, 1, 1],
            'C': [1, 1, 1, 1, 1, 1, 1]
        }
        self.df = pd.DataFrame(data)
        self.no_duplicates_df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})

    def test_fit(self):
        # Test that feature names are set correctly
        transformer = DuplicateValuesToNan()
        transformer.fit(self.df)
        self.assertEqual(transformer.feature_names_in_, ['A', 'B', 'C'])
        self.assertEqual(transformer.feature_names_out_, ['A', 'B', 'C'])
        self.assertEqual(transformer.value_to_replace, 0)
        self.assertEqual(transformer.n_max_duplicates, 144)

    def test_transform(self):
        # Test transformation with duplicate values
        transformer = DuplicateValuesToNan(0, 2)
        transformed_df = transformer.fit_transform(self.df)
        expected_output = {
            'A': [1, 2, 2, 3, 4, 5, 5],
            'B': [0, 0, np.nan, np.nan, np.nan, 1, 1],
            'C': [1, 1, 1, 1, 1, 1, 1]
        }
        pd.testing.assert_frame_equal(transformed_df, pd.DataFrame(expected_output))

        transformer = DuplicateValuesToNan(1, 2)
        transformed_df = transformer.fit_transform(self.df)
        expected_output = {
            'A': [1, 2, 2, 3, 4, 5, 5],
            'B': [0, 0, 0, 0, 0, 1, 1],
            'C': [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        pd.testing.assert_frame_equal(transformed_df, pd.DataFrame(expected_output))

        transformer = DuplicateValuesToNan(1, 1)
        transformed_df = transformer.fit_transform(self.df)
        expected_output = {
            'A': [1, 2, 2, 3, 4, 5, 5],
            'B': [0, 0, 0, 0, 0, 1, np.nan],
            'C': [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        pd.testing.assert_frame_equal(transformed_df, pd.DataFrame(expected_output))

    def test_no_duplicates(self):
        # Test transformation without duplicate values
        transformer = DuplicateValuesToNan(0, 2)
        x_transformed = transformer.fit_transform(self.no_duplicates_df)
        pd.testing.assert_frame_equal(x_transformed, self.no_duplicates_df)

    def test_exclude_features(self):
        # Test that excluded features are not transformed
        transformer = DuplicateValuesToNan(value_to_replace=0, n_max_duplicates=2, features_to_exclude=['C'])
        transformed_df = transformer.fit_transform(self.df)

        # Check that column 'C' remains unchanged
        expected_output = {
            'A': [1, 2, 2, 3, 4, 5, 5],
            'B': [0, 0, np.nan, np.nan, np.nan, 1, 1],
            'C': [1, 1, 1, 1, 1, 1, 1]  # This column should remain the same
        }
        pd.testing.assert_series_equal(transformed_df['C'], self.df['C'])
        pd.testing.assert_frame_equal(transformed_df, pd.DataFrame(expected_output))
