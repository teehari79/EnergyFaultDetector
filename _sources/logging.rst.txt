Logging Configuration
=====================

The framework uses Python's built-in logging module to provide logging capabilities. By default, the logging
configuration is defined in a YAML file. You can customize this configuration to suit your needs.

Default Configuration
---------------------

The framework uses a default logging configuration file named ``logging.yaml``.
The logger used throughout the code is called ``energy_fault_detector``.

The default logging configuration is as follows.

.. code-block:: yaml

    version: 1
    disable_existing_loggers: False
    formatters:
      simple:
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers:
      console:
        class: logging.StreamHandler
        level: DEBUG
        formatter: simple
        stream: ext://sys.stdout

    loggers:
      energy_fault_detector:
        level: INFO
        handlers: [console]
        propagate: no

    root:
      level: INFO
      handlers: [console]

You can silence the logger as follows:

.. code-block:: python
    import logging
    from energy_fault_detector.fault_detector import FaultDetector, Config

    logger = logging.getLogger('energy_fault_detector')
    logger.setLevel(logging.CRITICAL)


Customizing Logging
-------------------

You can specify your own logging configuration file by setting an environment variable before running the application.

To specify a custom logging configuration, set the ``LOG_CFG`` environment variable to the path of your configuration file:

.. code-block:: bash

    export LOG_CFG="/path/to/your_logging.yaml"

or
.. code-block:: python

    import os

    os.environ['LOG_CFG'] = '/path/to/your_logging.yaml'
