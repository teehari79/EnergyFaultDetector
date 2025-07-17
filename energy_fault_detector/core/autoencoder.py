
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, Tuple, List
import warnings
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# # pylint: disable=E0401,E0611,C0413
from tensorflow.keras.models import load_model as load_keras_model, Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, Callback

from energy_fault_detector.core.save_load_mixin import SaveLoadMixin

DataType = Union[np.ndarray, pd.DataFrame]
logger = logging.getLogger('energy_fault_detector')


class Autoencoder(ABC, SaveLoadMixin):
    """Autoencoder template. Compatible with sklearn and keras/tensorflow.

    Args:
        learning_rate: learning rate of the adam optimizer.
        batch_size: number of samples per batch.
        epochs: number of epochs to run.
        loss_name: name of loss metric to use.
        metrics: list of additional metrics to track.
        decay_rate: learning rate decay. Optional. If not defined, a fixed learning rate is used.
        decay_steps: number of steps to decay learning rate over. Optional.
        early_stopping: If True the early stopping callback will be used in the fit method. Early stopping will
            interrupt the training procedure before the last epoch is reached if the loss is not improving.
            The exact time of the interruption is based on the patience parameter.
        patience: parameter for early stopping. If early stopping is used the training will end if more than
            patience epochs in a row have not shown an improved loss.
        min_delta: parameter of the early stopping callback. If the losses of an epoch and the next epoch differ
            by less than min_delta, they are considered equal (i.e. no improvement).
        noise: float value that determines the influence of the noise term on the training input. High values mean
            highly noisy input. 0 means no noise at all. If noise >0 is used validation metrics will not be
            affected by it. Thus training loss and validation loss can differ depending on the magnitude of noise.
    """

    def __init__(self, learning_rate: float, batch_size: int, epochs: int,
                 loss_name: str, metrics: List[str], decay_rate: float, decay_steps: float,
                 early_stopping: bool, patience: int, min_delta: float, noise: float,
                 **kwargs):
        super().__init__()

        self.learning_rate = (
            ExponentialDecay(initial_learning_rate=learning_rate, decay_rate=decay_rate, decay_steps=decay_steps)
            if decay_rate and decay_steps else learning_rate
        )

        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.loss_name: str = loss_name
        self.metrics: List[str] = [] if metrics is None else metrics
        self.noise: float = noise

        self.model: Optional[KerasModel] = None
        self.encoder: Optional[KerasModel] = None
        self.history: Any = None

        self.callbacks: List[Callback] = [
            EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, restore_best_weights=True)
        ] if early_stopping else []

    def __call__(self, x: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """Calls the model on new inputs."""
        return self.model(x)

    @abstractmethod
    def create_model(self, input_dimension: Union[int, Tuple], **kwargs) -> KerasModel:
        """Create a keras model, sets the model and (optionally) encoder attributes.

        Args:
            input_dimension: number of features in input data.

        Returns:
            A Keras model.
        """

    def compile_model(self, new_learning_rate: float = None, **kwargs):
        """Compile (or recompile) model with Adam optimizer, optionally with a different learning rate."""

        if self.model is None:
            raise ValueError('You need to create the model first by calling the `create_model` method.')

        learning_rate = new_learning_rate if new_learning_rate else self.learning_rate
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss_name, metrics=self.metrics)

    def fit(self,
            x: DataType,
            x_val: DataType = None,
            **kwargs) -> 'Autoencoder':  # pylint: disable=W0237
        """Fit the autoencoder model.

        Args:
            x: training data
            x_val: validation data

        Returns:
            Fitted model.
        """

        if self.model is None:
            self.create_model(input_dimension=x.shape[1])

        self.compile_model()
        if 'callbacks' in kwargs:
            self.callbacks += kwargs['callbacks']
            kwargs.pop('callbacks')

        self._fit_model(x, x_val, batch_size=self.batch_size, epochs=self.epochs, callbacks=self.callbacks, **kwargs)
        return self

    def _fit_model(self, x: DataType, x_val: DataType, batch_size, epochs: int, callbacks: List[Callback],
                   **kwargs) -> None:

        fit_history = self.model.fit(
            self._apply_noise(x), x,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=None if x_val is None else (x_val, x_val),
            callbacks=callbacks,
            **kwargs
        )

        self._extend_fit_history(fit_history.history)

    def tune(self, x: DataType, x_val: DataType = None, learning_rate: float = 0.001, tune_epochs: int = 5,
             **kwargs) -> 'Autoencoder':
        """Tune full autoencoder by extending the model fitting process by tune_epochs.

        Args:
            x: training data
            x_val: validation data
            learning_rate: learning rate to use during tuning.
            tune_epochs: number of epochs to tune.
            kwargs: other keyword args for the keras `Model.fit` method.

        Returns:
            Tuned model.
        """

        self.compile_model(learning_rate)  # sets new learning rate
        self._fit_model(
            x, x_val,
            batch_size=self.batch_size,
            epochs=tune_epochs + self.epochs,
            callbacks=self.callbacks,
            initial_epoch=self.epochs,
            **kwargs
        )
        return self

    def tune_decoder(self, x: pd.DataFrame, x_val: pd.DataFrame = None, learning_rate: float = None,
                     tune_epochs: int = 5, **kwargs) -> 'Autoencoder':
        """Tune decoder only - weights of the encoder are unchanged. Weight tuning is done by extending the model
        fitting process by tune_epochs.

        Args:
            x: training data
            x_val: validation data
            learning_rate: learning rate to use during tuning. Default original learning rate.
            tune_epochs: number of epochs to tune.
            kwargs: other keyword args for the keras `Model.fit` method.

        Returns:
            Tuned model.
        """

        if self.encoder is None:
            raise ValueError("Encoder is not created.")

        self.encoder.trainable = False
        self.tune(x=x, x_val=x_val, learning_rate=learning_rate, tune_epochs=tune_epochs, **kwargs)
        return self

    def encode(self, x: DataType) -> np.ndarray:
        """Return latent representation using autoencoder."""
        if self.encoder is None:
            raise ValueError("Encoder is not created.")

        return self.encoder.predict(x)

    def predict(self, x: DataType, **kwargs) -> DataType:
        """Predict values using fitted model.

        Args:
            x: input data
        """

        if not self._is_fitted():
            raise ValueError(f'{self.__class__} object needs to be fitted first!')

        return self._predict(x, **kwargs)

    def _predict(self, x: DataType, **kwargs) -> DataType:
        """Predict with fitted model.

        Args:
            x: input data
            kwargs: other keyword args for the keras `Model.predict` method.

        Returns:
            AE reconstruction of the input data.
        """
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(self.model.predict(x, **kwargs), index=x.index, columns=x.columns)
        return self.model.predict(x, **kwargs)

    def get_reconstruction_error(self, x: DataType) -> DataType:
        """Get the reconstruction error: output - input.

        Args:
            x: input data

        Returns:
            AE reconstruction error of the input data.
        """

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(self.predict(x) - x, index=x.index, columns=x.columns)
        return self.predict(x) - x

    def save(self, directory: str, overwrite: bool = False, **kwargs):  # pylint: disable=W0221,W0613
        """Save the model object in given directory, filename is the class name.

        Args:
            directory: directory to save the object in.
            overwrite: whether to overwrite existing data, default False.
        """

        self._create_empty_dir(directory, overwrite)
        file_path = os.path.join(directory, self.__class__.__name__)

        if self.model is not None:
            self.model.save(file_path + '.model')

        if self.encoder is not None:
            self.encoder.save(file_path + '.encoder')

        ae_dict = self.__dict__.copy()
        ae_dict['model'] = None
        ae_dict['encoder'] = None
        file_name = file_path + '.attrs'
        with open(file_name, 'wb') as f:
            f.write(pickle.dumps(ae_dict))

    def load(self, directory: str, **kwargs) -> 'Autoencoder':  # pylint: disable=W0221,W0613
        """Load the model object from given directory."""

        file_name = os.path.join(directory, self.__class__.__name__)
        with open(file_name + '.attrs', 'rb') as f:
            class_data = f.read()

        self.__dict__ = pickle.loads(class_data)
        if os.path.exists(file_name + '.model'):
            self.model = load_keras_model(file_name + '.model')
            if os.path.exists(file_name + '.encoder'):
                self.encoder = load_keras_model(file_name + '.encoder')
        else:
            warnings.warn('No fitted model was found.')

        return self

    def _is_fitted(self) -> bool:
        """Check whether fit was called at least once."""

        if self.model is None or self.history is None:
            return False
        return True

    def _extend_fit_history(self, tune_history: Dict[str, List[Any]]) -> None:
        """Extend the fit history."""
        if self.history is None:
            self.history = tune_history
            return

        for k, _ in self.history.items():
            if k in tune_history:
                self.history[k] = self.history[k] + tune_history[k]

    def _apply_noise(self, x: DataType) -> DataType:
        """Apply random normal noise - for denoising AEs.

        Args:
            x: Input data, which can be a NumPy array or a Pandas DataFrame.

        Returns:
            The input data with noise applied.
        """

        if self.noise == 0:
            return x

        return x + self.noise * np.random.normal(loc=0., scale=1.0, size=x.shape)

    def _apply_noise_generator(self, x):
        """Apply random normal noise to generator inputs - for denoising AEs.
        We assume the generator outputs inputs, outputs tuples."""

        if self.noise == 0:
            for batch in x:
                yield batch
            return

        for batch_input, batch_output in x:
            noisy_input = []
            for seq in batch_input:
                noisy_seq = seq + np.random.normal(0, self.noise, seq.shape)
                noisy_input.append(noisy_seq)

            yield np.array(noisy_input), batch_output
