import os
from argparse import Namespace

import torchvision.transforms.v2 as transforms

from vehicle_reid.datasets import VRIC, VeRi


def match_dataset(args: Namespace, split: str):
    transform_train = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((args.height, args.width)),
    ])

    match args.dataset:
        case "vric":
            root = os.path.join(args.data_path, "vric")
            dataset = VRIC(root=root, split=split, transform=transform_train)
        case "veri":
            root = os.path.join(args.data_path, "veri")
            dataset = VeRi(root=root, split=split, transform=transform_train)
        case _:
            raise ValueError("TODO: other datasets, or a bug in the code not parsing dataset through argparse.")

    return dataset
