import os

import torchvision.transforms.v2 as transforms

from vehicle_reid.config import cfg
from vehicle_reid.datasets import VRIC, VeRi


def match_dataset(split: str):
    transform_train = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)),
    ])

    path = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    image_index = os.path.join(path, "image_index.json")
    label_index = os.path.join(path, "label_index.json")

    match cfg.DATASET.NAME:
        case "vric":
            root = os.path.join(cfg.DATASET.PATH, "vric")
            dataset = VRIC(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform_train)
        case "veri":
            root = os.path.join(cfg.DATASET.PATH, "VeRi")
            dataset = VeRi(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform_train)
        case _:
            raise ValueError("TODO: other datasets, or a bug in the code not parsing dataset through argparse.")

    return dataset

