import logging
import os
import sys

from vehicle_reid import args
from vehicle_reid.config import cfg

def configure_logger():
    stream_handler = logging.StreamHandler(sys.stdout)

    handlers = [stream_handler]

    if cfg.MISC.LOG_DIR:
        file_handler = logging.FileHandler(os.path.join(cfg.MISC.LOG_DIR, "log.txt"), mode="w")
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

def main():
    configure_logger()

    args.parse_command()
