
import unittest

import pandas as pd

from energy_fault_detector.data_preprocessing.data_clipper import DataClipper


class TestDataClipper(unittest.TestCase):

    def setUp(self):
        self.data_clipper = DataClipper(lower_percentile=0.2, upper_percentile=0.8,
                                        features_to_exclude=['angle1', 'angle2'])

    def test_fit(self):
        x_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        self.data_clipper.fit(x_train)
        self.assertEqual(self.data_clipper.feature_names_in_, ['feature1', 'feature2'])
        self.assertEqual(self.data_clipper.feature_names_out_, ['feature1', 'feature2'])

    def test_transform(self):
        x_test = pd.DataFrame(
            {'feature1': [1, 2, 3, 4, 5], 'feature2': [4, 5, 6, 7, 8], 'angle1': [0, 45, 90, 135, 180],
             'angle2': [0, 45, 90, 135, 180]}
        )
        expected_output = pd.DataFrame(
            {'feature1': [1.8, 2, 3, 4, 4.2], 'feature2': [4.8, 5, 6, 7, 7.2], 'angle1': [0, 45, 90, 135, 180],
             'angle2': [0, 45, 90, 135, 180]}
        )
        self.data_clipper.fit(x_test)
        self.assertTrue(self.data_clipper.transform(x_test).equals(expected_output))
