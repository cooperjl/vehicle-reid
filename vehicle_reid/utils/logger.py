import logging
import os
from datetime import datetime

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except Exception:
            self.handleError(record)


def configure_logger(log_dir: str = None, config_name: str = None):
    """
    Configure the logging module, to setup printing the log to stdout, handle the log with tqdm, and
    save the log to an appropriately named file.

    Parameters
    ----------
    log_dir : str, optional
        Directory to save the log to. If not specified, do not save a log. In this case, logging still needs to
        be setup, just the file handler will not be enabled.
    config_name : str, optinal
        Name of the config file used, to be used in the name of the log file. If not specified, just defaults to
        "log" instead.
    """
    stream_handler = TqdmHandler()

    handlers = [stream_handler]

    if log_dir:
        name = config_name if config_name else "log"
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")

        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}-{timestamp}.txt"), mode="w"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
    )
