from argparse import Namespace

from vehicle_reid import args
from vehicle_reid.model import Model


def parse_arguments():
    parser = args.add_subparser(name="train", help="train the model")
    # parser.add_argument('dataset', metavar='dataset',
    #                     choices=args.DATASETS,
    #                     help='the name of the dataset to compute')
    parser.set_defaults(func=main)


def train_one_epoch(model: Model):
    """
    Singular epoch training function. TODO
    """

    model.train()


def main(args: Namespace):
    print("Train called")


if __name__ == "__main__":
    main()

