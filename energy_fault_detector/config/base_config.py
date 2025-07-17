"""Base config class template"""

import os
import logging
from typing import Dict, Optional, List, Any, Callable
import datetime
from abc import ABC

import yaml
from cerberus import Validator

logger = logging.getLogger('energy_fault_detector')

SCHEMA = {'allow_unknown': True}


class InvalidConfigFile(Exception):
    """Raise when config file is not valid"""


class BaseConfig(ABC):
    """Base configuration class.
    either config_filename or config_dict must be provided.
    """

    def __init__(self, config_filename: str = None, config_dict: Dict[str, Any] = None):

        self.configuration_file: str = config_filename
        self.config_dict: Optional[Dict] = config_dict

        # config schema - define in the subclasses
        self._schema: Dict = None
        self._extra_validation_checks = None
        # call self.read_config()

    def __getitem__(self, item) -> Any:
        return self.config_dict[item]

    def __contains__(self, item):
        return item in self.config_dict.keys()

    def __repr__(self):
        return self.config_dict.__repr__()

    def read_config(self, part: str = None) -> None:
        """Read and validate the configuration."""

        if self.configuration_file is not None:
            with open(self.configuration_file, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.safe_load(f)

        if self.config_dict is None:
            raise InvalidConfigFile('The configuration file is empty!')

        # backwards compatibility
        if part is not None:
            if part in self.config_dict:
                self.config_dict = self.config_dict[part]

        if self._schema is None:
            raise ValueError('The `_schema` attribute is missing...')

        self._validate_config(self.config_dict, schema=self._schema, extra_checks=self._extra_validation_checks)

    def update_config(self, new_config_dict: Dict[str, Any]) -> None:
        """Update the configuration. Sets the configuration_file path to None.

        Args:
            new_config_dict: dictionary with the new configuration.
        """

        new_config = self.config_dict.copy()
        new_config.update(new_config_dict)
        self._validate_config(new_config, self._schema, extra_checks=self._extra_validation_checks)
        self.configuration_file = None

    def write_config(self, file_name: Optional[str] = None, overwrite: bool = False) -> None:
        """Write the configuration to a yaml file"""

        if file_name is None and self.configuration_file is None:
            raise ValueError('No file name given and no known configuration file to overwrite.')

        file_name = file_name if file_name is not None else self.configuration_file
        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(f'File {file_name} already exists and overwrite is set to False.')

        sampler_params = self.config_dict.get('train', {}).get('autoencoder', {}).get('time_series_sampler', {})
        ts_freq = sampler_params.get('ts_freq')
        if ts_freq is not None:
            self.config_dict['train']['autoencoder']['time_series_sampler']['ts_freq'] = repr(ts_freq)

        # remove sampler from autoencoder params dictionary
        ts_sampler = self.config_dict.get('train', {}).get('autoencoder', {}).get('params', {}).get('ts_sampler')
        if ts_sampler is not None:
            self.config_dict['train']['autoencoder']['params'].pop('ts_sampler')

        with open(file_name, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config_dict, f)

    @staticmethod
    def _parse_dates(dates: List, dt_format: str = '%Y-%m-%d') -> List:
        """Parse given string dates"""

        parsed = []
        for date in dates:
            if isinstance(date, str):
                date = datetime.datetime.strptime(date, dt_format).date()
            elif isinstance(date, datetime.date):
                pass
            else:
                raise ValueError('Given date %s of type %s cannot be parsed' % date, type(date))
            parsed.append(date)

        return parsed

    def _validate_config(self, config_dict: Dict[str, Any], schema: Dict[str, str],
                         extra_checks: List[Callable[[Dict], None]] = None) -> None:
        """Check whether the provided configuraton has all expected keys.
        If the given file is a valid configuration, the config_dict attribute is set.

        Args:
            config_dict: dictionary containing the configuration
            schema: Cerberus schema to validate against
            extra_checks: list of extra validation checks to run. Should be functions that take the configuration
               dictionary as input and return the (updated) config_dict or raise InvalidConfigFile
        """

        validator = Validator(allow_unknown=True)  # allow_unknown for backwards compat
        if not validator.validate(config_dict, schema):
            raise InvalidConfigFile(f'Configuration is not valid: {validator.errors}')

        for key in list(config_dict.keys()):
            if key not in schema:
                logger.info('Key `%s` is an unknown field and will be ignored.', key)
                config_dict.pop(key)

        if config_dict == {}:
            raise InvalidConfigFile(f'The configuration file is empty for {self.__class__.__name__}.')

        config_dict = validator.normalized(config_dict)

        if extra_checks is not None:
            for extra_check in extra_checks:
                config_dict = extra_check(config_dict)

        self.config_dict = config_dict
