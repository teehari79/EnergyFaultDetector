"""Logging settings"""

import os
import logging.config as logging_config

import yaml


def setup_logging(default_path: str = 'logging.yaml', env_key: str = 'LOG_CFG') -> None:
    """Setup logging configuration

    Args:
        default_path (str): default logging configuration file. Default is 'logging.yaml'
        env_key (str): Environment variable holding logging config file path (overrides default_path). Default is
            'LOG_CFG'
    """

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    try:
        with open(path, 'rt', encoding='utf-8') as f:
            config = yaml.safe_load(f.read())
            # check paths exist or create them:
            for _, handler in config['handlers'].items():
                if handler.get('filename'):
                    dirname = os.path.dirname(handler['filename'])
                    if dirname != '' and not os.path.exists(dirname):
                        os.makedirs(dirname)

        logging_config.dictConfig(config)
    except Exception as e:
        raise ValueError(f"Error setting up logging: {e}") from e
