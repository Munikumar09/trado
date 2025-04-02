"""
This module provides utility functions for logging.

This module contains a function `get_logger` that returns a logger
object configured with a stream handler and a file handler. The logger
can be used to log messages with different log levels.

Example usage:
    logger = get_logger(__name__, log_level="INFO", log_to_file=True)
    logger.info("This is an info message")
    logger.warning("This is a warning message")
"""

import logging
from pathlib import Path


def get_logger(
    name: str, log_level: str = "INFO", log_to_file: bool = False
) -> logging.Logger:
    """
    Return a logger object configured with a stream handler and a file handler.

    Parameters
    ----------
    name: ``str``
        The name of the logger, this is usually the name of the module
    log_level: ``str``, ( default = "INFO" )
        The log level for the logger. The default value is "INFO"
    log_to_file: ``bool``, ( default = False )
        A boolean value to indicate if the logs should be written to a file.
        The file name will be the name of the module with a .log extension.

    Returns
    -------
    logger: ``Logger``
        The logger object configured with a stream handler and a file handler
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # Disable propagation to avoid duplicate logs

    # Add a stream handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_format = logging.Formatter(
        "%(asctime)s - %(name)s:%(lineno)d  - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # Add a file handler if log_to_file is True
    if log_to_file:
        log_file = f"{Path(name).stem}.log"
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(log_level)
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s:%(lineno)d  - %(levelname)s - %(message)s"
        )
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    # Suppress Kafka logs
    logging.getLogger("kafka").setLevel(
        logging.WARNING
    )  # Set to ERROR or CRITICAL to suppress more

    return logger
