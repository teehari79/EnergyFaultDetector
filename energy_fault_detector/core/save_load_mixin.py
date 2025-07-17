
import os
import shutil
import pickle


class SaveLoadMixin:
    """Save and load methods for model-like (regressors/classifiers/transformers) objects.

    Override the save and load functions when one of the attributes cannot be pickled, such as keras/tensorflow models.
    """

    def __init__(self, **kwargs):
        # **kwargs to ensure this class works fine in multiple inheritance cases.
        pass

    def save(self, directory: str, overwrite: bool = False, file_name: str = None, create_dir: bool = True):
        """Save the model object in given directory, filename is the class name.

        Args:
            directory: directory to save the object in.
            overwrite: whether to overwrite existing data, default False.
            file_name: name of the file to save the object in, if none, take the class name.
            create_dir: whether to create the directory, default True.
        """
        # Note: override when using Keras/Tensorflow, as those models cannot be pickled.
        if create_dir:
            self._create_empty_dir(directory, overwrite)

        if file_name is None:
            path = os.path.join(directory, self.__class__.__name__ + '.pkl')
        else:
            path = os.path.join(directory, file_name)
        with open(path, 'wb') as f:
            f.write(pickle.dumps(self.__dict__))

    def load(self, directory: str, file_name: str = None):
        """Load the model object from given directory."""
        # Note: override when using Keras/Tensorflow, as those models cannot be pickled.
        if file_name is None:
            path = os.path.join(directory, self.__class__.__name__ + '.pkl')
        else:
            path = os.path.join(directory, file_name)
        with open(path, 'rb') as f:
            class_data = f.read()

        self.__dict__ = pickle.loads(class_data)

    @staticmethod
    def _create_empty_dir(directory: str, overwrite: bool):
        """Create new directory and empty if it already exists

        Args:
            directory: directory to create.
            overwrite: whether to overwrite existing data, default False.
        """

        if os.path.exists(directory) and overwrite:
            shutil.rmtree(directory)
        elif os.path.exists(directory) and not overwrite:
            raise FileExistsError(f'Directory {directory} already exists!')
        os.makedirs(directory)
