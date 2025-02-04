import argparse
import os

from dataclasses import dataclass

import vehicle_reid.gms as gms
import vehicle_reid.test as test
import vehicle_reid.train as train

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(required=True, metavar="command")

@dataclass
class DEFAULTS:
    width: int = 224
    height: int = 224
    data_path: str = "data"

COMMANDS = [
    gms,
    test,
    train,
]

def parse_command():
    for command in COMMANDS:
        command.parse_arguments()

    args = parse()
    args.func(args)

def parse():
    return parser.parse_args()

def add_subparser(name: str, help: str):
    return subparsers.add_parser(name=name, help=help)

def dir_path(path):
    """Wrapper function to check whether the given path is valid"""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"invalid path: {path}")

