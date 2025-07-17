
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from energy_fault_detector.data_splitting.data_splitter import BlockDataSplitter


class TestDataSplitter(TestCase):
    def setUp(self) -> None:
        self.data_splitter = BlockDataSplitter(train_block_size=6, val_block_size=3)

        # input
        self.data = np.array(range(20))
        self.normal_index = np.array([True]*10 + [False]*4 + [True]*6)

    def test_split(self) -> None:
        # expected output
        exp_train_data = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19])
        exp_val_data = np.array([6, 7, 8, 15, 16, 17])
        exp_train_normal_index = np.array([True] * 7 + [False] * 4 + [True] * 3)
        exp_val_normal_index = np.array([True] * 6)

        train_data, val_data, train_normal_index, val_normal_index = self.data_splitter.split(self.data,
                                                                                              self.normal_index)

        assert_array_equal(train_data, exp_train_data)
        assert_array_equal(val_data, exp_val_data)
        assert_array_equal(train_normal_index, exp_train_normal_index)
        assert_array_equal(val_normal_index, exp_val_normal_index)

    def test_split_x(self) -> None:
        # expected output
        exp_train_data = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19])
        exp_val_data = np.array([6, 7, 8, 15, 16, 17])

        train_data, val_data = self.data_splitter.split(x=self.data)

        assert_array_equal(train_data, exp_train_data)
        assert_array_equal(val_data, exp_val_data)
