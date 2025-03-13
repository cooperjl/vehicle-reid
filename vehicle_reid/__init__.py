import logging
import os
import sys

from vehicle_reid import args
from vehicle_reid.config import cfg


def main():
    #formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    #stream_handler.setFormatter(formatter)
    #logger.addHandler(stream_handler)

    handlers = [stream_handler]

    if cfg.MISC.LOG_DIR:
        file_handler = logging.FileHandler(os.path.join(cfg.MISC.LOG_DIR, "log.txt"), mode="w")
        #file_handler.setLevel(logging.DEBUG)
        #file_handler.setFormatter(formatter)
        #logger.addHandler(file_handler)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    args.parse_command()
