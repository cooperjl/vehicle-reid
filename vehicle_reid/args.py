import argparse
import os
from dataclasses import dataclass, fields

from vehicle_reid import gms, test, train
from vehicle_reid.datasets import visualise

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(required=True, metavar="command")

@dataclass
class DEFAULTS:
    width: int = 224
    height: int = 224
    data_path: str = "data"
    gms_path: str = "gms"

DATASETS = [
    "vric",
    "vehicle_id",
    "veri",
]

COMMANDS = [
    gms,
    test,
    train,
    visualise,
]

def parse_command():
    for command in COMMANDS:
        command.parse_arguments()

    parser.set_defaults(
        width=DEFAULTS.width, 
        height=DEFAULTS.height,
        data_path=DEFAULTS.data_path,
        gms_path=DEFAULTS.gms_path,
    )

    args = parse()
    args.func(args)

def parse():
    return parser.parse_args()

def add_subparser(name: str, help: str):
    return subparsers.add_parser(name=name, help=help)

def dir_path(path):
    """Wrapper function to check whether the given path is a valid directory."""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"invalid path: {path}")

