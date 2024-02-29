# Copyright Institute for Data Intensive Engineering and Science
# License: MIT

import logging
import time


class LogTime:
    """A context manager class for logging code timings and errors.

    This class will log the time it takes to run some code along with any
    exceptions that occur within in the context. Timings are logged at the INFO
    level and errors are logged at the FATAL level.

    Example:

    >>> import logging
    >>> logger = logging.getLogger("example")
    >>>
    >>> with LogTime(logger, "Calculating Sum"):
    >>>     a = 2 + 2

    The log file will look something like (depending on your formatting):

    >>> Starting Calculating Sum
    >>> Completed in 7.62939453125e-06 seconds
    """

    def __init__(self, logger: logging.Logger, task_str: str):
        """A context manager class for logging code timings and errors.

        Args:
            logger (logging.Logger): The logger instance to use for logging
            task_str (str): The string describing the section of code being run

        """
        self.logger = logger
        self.task_str = task_str

    def __enter__(self):
        self.logger.info(f"Starting {self.task_str}")
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.logger.info(f"Completed in {time.time() - self.start} seconds")
        else:
            self.logger.fatal(f"{exc_type}\n{exc_value}\n{traceback}")


class DefaultLogTime:
    """Same as LogTime but just prints. Useful for debugging"""

    def __init__(self, logger: logging.Logger, task_str: str):
        self.logger = logger
        self.task_str = task_str

    def __enter__(self):
        print(f"Starting {self.task_str}")
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            print(f"Completed in {time.time() - self.start} seconds")
        else:
            print(f"{exc_type}\n{exc_value}\n{traceback}")


class DontLogTime:
    """Same as LogTime, but actually not. It does nothing"""

    def __init__(self, logger: logging.Logger, task_str: str):
        self.logger = logger
        self.task_str = task_str

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
