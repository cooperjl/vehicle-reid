import argparse
import logging
import os

from vehicle_reid import gms, test, train
from vehicle_reid.config import cfg
from vehicle_reid.datasets import visualise
from vehicle_reid.utils import configure_logger, normal_values, set_seed

COMMANDS = {
    "gms": gms.main,
    "test": test.main,
    "train": train.train,
    "visualise": visualise.visualise_dataset,
    "normalise": normal_values.calculate_normal_values,
}

parser = argparse.ArgumentParser(description="Vehicle Re-Identification using Deep Learning")

def parse_command():
    """
    Function which parses the command line arguments, and loads the configuration file specified,
    as well as choosing the command to run.
    """
    parser.add_argument("--config-file", type=file_path,
                        default="configs/veri.yml",
                        help="path to config file")

    parser.add_argument("command",
                        choices=COMMANDS.keys())

    parser.add_argument("config",
                        default=None,
                        help="override config options from the command line",
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
        
        configure_logger(cfg.MISC.LOG_DIR, os.path.splitext(os.path.basename(args.config_file))[0])
        logger = logging.getLogger(__name__)
        logger.info(f"loaded configuration file: {args.config_file}")
    else:
        configure_logger()

    cfg.merge_from_list(args.config)
    cfg.freeze()

    set_seed()

    # execute the command obtained from the dict
    COMMANDS[args.command]()

def dir_path(path):
    """Wrapper function to check whether the given path is a valid directory."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"invalid path: {path}")

def file_path(path):
    """Wrapper function to check whether the given path is a valid file."""
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"invalid path: {path}")

