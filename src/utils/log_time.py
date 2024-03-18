# MIT License

# Copyright (c) 2024 The Johns Hopkins University, Institute for Data Intensive Engineering and Science

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import time
from typing import Optional


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

    def __init__(self, task_str: str, logger: Optional[logging.Logger] = None):
        """A context manager class for logging code timings and errors.

        Args:
            task_str (str): The string describing the section of code being run
            logger (logging.Logger): The logger instance to use for logging

        """
        self.logger = logger if logger else PrintLogger("default_logger")
        self.task_str = task_str

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.logger.info(f"Completed {self.task_str} in {time.time() - self.start} seconds")
        else:
            self.logger.fatal(f"{exc_type}\n{exc_value}\n{traceback}")


class PrintLogger(logging.Logger):
    """A simple logger that prints to stdout."""

    def __init__(self, name: str):
        super().__init__(name)

    def _log(self, level, msg, args, exc_info=None, extra=None):
        print(msg % args)


# https://stackoverflow.com/a/11233293/2691018
def setup_logger(name: str, log_file: str, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
