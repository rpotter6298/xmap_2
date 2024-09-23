"""
logbook.py

This module provides a function to set up a logger for different classes in an application.
The logger is configured to write logs to a file, with the file path specified either as a string
or a pathlib.Path object. If the path provided is a relative path, it is assumed to be relative
to the 'logs' directory. If the specified path does not exist, the module will create the necessary
directories.

Functions:
    setup_class_logger(class_name, log_path): Configures and returns a logger object for the given class name.
        The logger writes messages to a log file at the specified path. If the path is a string or a Path object
        that is not within the 'logs' directory, it will be adjusted to be so, and the corresponding directory
        structure will be created if necessary.

Example:
    logger = setup_class_logger('MyClass', 'subdir/myclass.log')
    or
    logger = setup_class_logger('MyClass', Path('subdir/myclass.log'))
    This will create a logger for 'MyClass' that writes to 'logs/subdir/myclass.log'.
"""

import logging
from pathlib import Path

# Define the base directory for logs
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def setup_class_logger(class_name, log_path):
    """
    Sets up a logger with the specified name and path for log files.

    Args:
        class_name (str): The name of the class for which the logger is being created.
        log_path (str | Path): The path indicating where the log file should be stored.
                            If the path is a string, it will be converted to a Path object.
                            If the path is outside the 'logs' directory, it will be adjusted accordingly.

    Returns:
        logging.Logger: A logger instance configured to write logs to the specified file.
    """
    # Convert log_path to Path object if it's a string
    if isinstance(log_path, str):
        log_path = Path(log_path)

    # Ensure the log_path is relative to logs_dir
    if log_path.is_absolute() or logs_dir not in log_path.parents:
        log_path = logs_dir / log_path

    # Create the directories if they don't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a logger with the given class name
    logger = logging.getLogger(class_name)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)

    return logger
