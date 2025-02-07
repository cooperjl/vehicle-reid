from argparse import Namespace
import os

from vehicle_reid import args, gms
from vehicle_reid.model import Model


def parse_arguments():
    parser = args.add_subparser(name="train", help="train the model")
    parser.add_argument('dataset', metavar='dataset',
                        choices=args.DATASETS,
                        help='the name of the dataset to train on')
    parser.add_argument('--gms-path', type=args.dir_path,
                        default='gms',
                        help='path to load gms matches from (default: %(default)s)')
    parser.set_defaults(func=main)


def train_one_epoch(model: Model):
    """
    Singular epoch training function. TODO
    """

    model.train()


def mine_triplet(gms_path: str):
    gms_matrices = gms.load_data(gms_path)

    print(gms_matrices)


def main(args: Namespace):
    print("Train called")

    gms_path = os.path.join(args.gms_path, args.dataset)
    mine_triplet(gms_path)



