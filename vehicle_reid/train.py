import os
from argparse import Namespace

import torch

from vehicle_reid import args, gms
from vehicle_reid.datasets import load_data
from vehicle_reid.model import Model


def parse_arguments():
    parser = args.add_subparser(name="train", help="train the model")
    parser.add_argument('dataset', metavar='dataset',
                        choices=args.DATASETS,
                        help='the name of the dataset to train on')
    parser.add_argument('--data-path', type=args.dir_path,
                        default=args.DEFAULTS.data_path,
                        help='path where the datasets are stored (default: %(default)s)')
    parser.add_argument('--gms-path', type=args.dir_path,
                        default=args.DEFAULTS.gms_path,
                        help='path to load gms matches from (default: %(default)s)')
    parser.add_argument('--epochs', type=int,
                        required=True,
                        help='number of epochs for training')
    parser.add_argument('--width', type=int,
                        default=args.DEFAULTS.width,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--height', type=int,
                        default=args.DEFAULTS.height,
                        help='width to resize images to (default: %(default)s)')
    parser.set_defaults(func=train)


def train_one_epoch(model: Model, dataloader, gms_dict: dict, batch_size: int, width: int, height: int):
    """
    Singular epoch training function. TODO
    """

    model.train()
    
    trainX = torch.zeros((batch_size * 3, 3, height, width), dtype=torch.float32)
    trainY = torch.zeros((batch_size * 3), dtype=torch.float32)

    for image, labels in dataloader:
        #mine_triplets(batch_size, gms_dict)
        print(type(labels))

def mine_triplets(batch_size: int, gms_dict: dict, labels: torch.Tensor):
    # TODO: im quite sure theres a better way to do this, than iterating through the batch size....
    for i in range(batch_size):
        labelx = str(labels[i])
        indexx


    


def train(args: Namespace):
    print("Train called")

    gms_path = os.path.join(args.gms_path, args.dataset)
    gms_dict = gms.load_data(gms_path)

    model = Model()
    dataloader = load_data(args, "train")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}:")

        train_one_epoch(model, dataloader, gms_dict, 16, 224, 224)

