import logging
import os

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console
    
    Credit: https://stackoverflow.com/a/67257516
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except Exception:
            self.handleError(record)


def configure_logger(log_dir: str=None):
    stream_handler = TqdmHandler()

    handlers = [stream_handler]

    if log_dir:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode="w")
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

