import os
from argparse import Namespace

import torchvision.transforms.v2 as transforms

from vehicle_reid.datasets import VRIC, VeRi


def match_dataset(args: Namespace, split: str):
    transform_train = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((args.height, args.width)),
    ])

    path = os.path.join(args.gms_path, args.dataset)
    image_index = os.path.join(path, "image_index.json")
    label_index = os.path.join(path, "label_index.json")

    match args.dataset:
        case "vric":
            root = os.path.join(args.data_path, "vric")
            dataset = VRIC(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform_train)
        case "veri":
            root = os.path.join(args.data_path, "VeRi")
            dataset = VeRi(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform_train)
        case _:
            raise ValueError("TODO: other datasets, or a bug in the code not parsing dataset through argparse.")

    return dataset

