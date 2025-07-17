
from unittest import TestCase

import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from energy_fault_detector.data_preprocessing.angle_transformer import AngleTransformer


class TestAngleTransformer(TestCase):
    def setUp(self) -> None:
        self.angle_transformer = AngleTransformer(angles=['Sensor_1'])

        # generate data
        length = 10  # choose an even number for simplicity
        time_index = pd.date_range(start='1/1/2021', end='10/1/2021', periods=length)
        data = {'Sensor_1': list(range(length)),
                'Sensor_2': [np.nan] + list(range(1, length)),
                'Sensor_3': list(range(int(length / 2))) + [np.nan] * int(length / 2),
                'Sensor_4': [0.] + [np.nan] * (length - 1),
                'Sensor_5': [0.] * length}

        # input
        self.test_data = pd.DataFrame(index=time_index, data=data)

    def test_fit(self):
        # expected fit result
        expected_attributes = [
            'Sensor_3',
            'Sensor_4',
            'Sensor_2',
            'Sensor_5',
            'Sensor_1_cosine',
            'Sensor_1_sine'
        ]

        self.angle_transformer.fit(self.test_data)
        attributes = self.angle_transformer.feature_names_out_
        self.assertSetEqual(set(expected_attributes), set(attributes))

    def test_transform(self):
        # expected result
        expected_sines = np.sin(self.test_data['Sensor_1'] * np.pi / 180.)
        expected_cosines = np.cos(self.test_data['Sensor_1'] * np.pi / 180.)

        df = self.angle_transformer.fit_transform(self.test_data)
        assert_array_almost_equal(expected_sines.values, df['Sensor_1_sine'].values)
        assert_array_almost_equal(expected_cosines.values, df['Sensor_1_cosine'].values)

    def test_inverse_transform(self):
        self.angle_transformer.fit(self.test_data)
        assert_frame_equal(
            self.angle_transformer.inverse_transform(self.angle_transformer.transform(self.test_data)).astype(float),
            self.test_data.astype(float)
        )

    def test_multiple_ranges(self):
        data = {'s1': np.arange(0, 360, 45),
                's2': np.arange(-180, 180, 45),
                's3': [-90, -60, -30, 0, 30, 60, 80, 90],
                's4': np.arange(-400, 400, 100)}
        test_df = pd.DataFrame(data)
        at = AngleTransformer(angles=['s1', 's2', 's3', 's4'], trust_bad_angles=False)
        df_transformed = at.fit_transform(test_df)
        inv_df = at.inverse_transform(df_transformed)

        self.assertDictEqual(at.ranges_,
                             {'s1': (0, 360), 's2': (-180, 180), 's3': (-180, 180)})
        assert_frame_equal(inv_df.astype(float), test_df[['s1', 's2', 's3']].astype(float))

        at = AngleTransformer(angles=['s1', 's2', 's3', 's4'], trust_bad_angles=True)
        df_transformed = at.fit_transform(test_df)
        inv_df = at.inverse_transform(df_transformed)

        self.assertDictEqual(at.ranges_,
                             {'s1': (0, 360), 's2': (-180, 180), 's3': (-180, 180), 's4': (0, 360)})
        expected_result = np.array([320.0, 60.0, 160.0, 260.0, 0.0, 100.0, 200.0, 300.0])
        assert_array_almost_equal(expected_result, inv_df['s4'].values, decimal=12)
