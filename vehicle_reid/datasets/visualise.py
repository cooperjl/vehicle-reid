import argparse

import torch
import matplotlib.pyplot as plt

from vehicle_reid import args
from vehicle_reid.datasets import VehicleReIdDataset, match_dataset

def parse_arguments():
    parser = args.add_subparser(name="visualise", help="visualise a random sample of the given dataset")
    parser.add_argument('dataset', metavar='dataset',
                        choices=args.DATASETS,
                        help='the name of the dataset to visualise')
    parser.add_argument('--data-path', type=args.dir_path,
                        default=args.DEFAULTS.data_path,
                        help='path where the datasets are stored (default: %(default)s)')
    parser.set_defaults(func=visualise_dataset)

def visualise(dataset: VehicleReIdDataset):
    """Basic visualisation for the dataset

    Note: do not use the raw VehicleReIdDataset type.
    """

    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.tight_layout()
        ax.axis('off')

        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label, _ = dataset[sample_idx]
        img = img.permute(1, 2, 0)
        plt.title(label)
        plt.imshow(img)

    plt.show()

def visualise_dataset(args: argparse.Namespace):
    dataset = match_dataset(args, "train")
    visualise(dataset)

