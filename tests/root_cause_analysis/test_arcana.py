import os
from unittest import TestCase
import pickle as pkl

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Input
import tensorflow as tf

from energy_fault_detector.root_cause_analysis.arcana import Arcana
from energy_fault_detector.autoencoders import MultilayerAutoencoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestArcana(TestCase):

    def setUp(self) -> None:
        self.n = 1000
        data = np.array([[
            x,
            (x + 1),
            x ** 2,
            x * 3,
            x * 2 + 5,
            np.exp(x / self.n)
        ] for x in range(self.n)])
        self.data = (data - data.mean(axis=0)) / data.std(axis=0)
        time_index = pd.date_range(start="01-01-2022", periods=self.n, freq='10min')
        self.data_frame = pd.DataFrame(index=time_index, data=self.data)  # find_arcana_bias expects a pandas Dataframe

        self.ml_ae = MultilayerAutoencoder()
        input_dim = self.data.shape[1]
        ml_input_layer = Input(shape=(input_dim,))
        encoded = Dense(10, input_shape=(input_dim,), activation="linear")(ml_input_layer)
        decoded = Dense(input_dim, activation="linear")(encoded)
        ml_model = Model(inputs=ml_input_layer, outputs=decoded)
        ml_model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_absolute_error'])
        ml_model.load_weights(os.path.join(PROJECT_ROOT, 'test_data/ml_model_weights.h5'))
        self.ml_ae.model = ml_model
        self.ml_ae.history = 0  # ml_ae._is_fitted() will now return True so predict can be used without fit

    def test_find_arcana_bias(self):
        with open(os.path.join(PROJECT_ROOT, 'test_data/arcana_bias.pkl'), 'rb') as file:
            expected_bias = pkl.load(file)

        arcana = Arcana(model=self.ml_ae, num_iter=42)
        bias, _, _ = arcana.find_arcana_bias(self.data_frame)
        assert_array_almost_equal(expected_bias, bias.values, decimal=3)
        self.assertIsInstance(bias, pd.DataFrame)

    def test_find_arcana_bias_with_history(self):
        arcana = Arcana(model=self.ml_ae, num_iter=51)  # at least 50 iterations are needed
        bias, losses, bias_history = arcana.find_arcana_bias(self.data_frame, track_losses=True, track_bias=True)
        self.assertIsInstance(bias_history, list)
        self.assertTrue(len(bias_history) == 3)  # init bias + bias of 1st iteration and bias of 50th iteration
        self.assertIsInstance(losses, pd.DataFrame)
        for loss in losses:
            self.assertTrue(len(losses[loss]) == 2)  # losses of 1st iteration and 50th iteration

    def test_decreasing_loss(self):

        for alpha in [0, 0.99]:
            arcana = Arcana(model=self.ml_ae, num_iter=1, alpha=alpha, init_x_bias='recon')
            bias = arcana.initialize_x_bias(self.data.astype('float32'))
            last_loss = 1e8
            x = tf.Variable(self.data, dtype=tf.float32)
            bias = tf.Variable(bias, dtype=tf.float32)
            for _ in range(5):
                bias, losses, _ = arcana.update_x_bias(x=x, x_bias=bias)
                self.assertLess(losses[0].numpy(), last_loss)
                last_loss = losses[0].numpy()

    def test_init_bias(self):
        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='weightedB', alpha=0.6)
        bias_expected = 0.6 * (self.ml_ae.predict(self.data) - self.data)
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='weightedA', alpha=0.6)
        bias_expected = 0.4 * (self.ml_ae.predict(self.data) - self.data)
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='recon', alpha=0.6)
        bias_expected = self.ml_ae.predict(self.data) - self.data
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

        arcana = Arcana(model=self.ml_ae, num_iter=42, init_x_bias='zero', alpha=0.6)
        bias_expected = 0 * self.data
        bias = arcana.initialize_x_bias(self.data)
        assert_array_almost_equal(bias_expected, bias)

    def test_draw_samples(self):
        arcana = Arcana(model=self.ml_ae, num_iter=42, max_sample_threshold=self.n)
        selection = arcana.draw_samples(x=self.data)
        self.assertTupleEqual(self.data.shape, self.data[selection].shape)

        arcana.max_sample_threshold = self.n - 1
        selection = arcana.draw_samples(x=self.data)
        self.assertTupleEqual(self.data[:-1].shape, self.data[selection].shape)

    def test_ignore_features(self):
        ignore_cols = [self.data_frame.columns[0], 'non_existing_feature']
        arcana = Arcana(model=self.ml_ae, num_iter=5, ignore_features=ignore_cols)
        subset = self.data_frame.iloc[:20]
        bias, losses, _ = arcana.find_arcana_bias(subset, track_losses=True)
        self.assertTrue((bias[ignore_cols[0]] == 0).all())
        # ensure optimisation still runs and logs losses
        self.assertFalse(losses.empty)

