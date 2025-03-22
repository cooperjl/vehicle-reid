import os

import torch
import torchvision.transforms.v2 as transforms

from vehicle_reid.datasets import VRIC, VeRi
from vehicle_reid.config import cfg

def transform_train():
    return transforms.Compose([
        transforms.Resize(cfg.INPUT.WIDTH + int(cfg.INPUT.WIDTH * 1/8), antialias=True),
        transforms.RandomCrop(size=(cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT)),
        transforms.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD),
    ])

def transform_test(normalise=True):
    transform = [
        transforms.Resize(size=cfg.INPUT.WIDTH, antialias=True),
        transforms.CenterCrop(size=(cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT)),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
    if normalise:
        transform.append(transforms.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD))

    return transforms.Compose(transform)


def match_dataset(split: str):
    path = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    image_index_path = os.path.join(path, "image_index.json")
    label_index_path = os.path.join(path, "label_index.json")

    image_index = image_index_path if os.path.isfile(image_index_path) else None
    label_index = label_index_path if os.path.isfile(label_index_path) else None

    match split:
        case "train":
            transform = transform_train()
        case "normal": # used for calculating normalise values
            transform = transform_test(normalise=False) # remove normalise from transform and don't augment
            split = "train" # use train split
        case _:
            transform = transform_test()

    match cfg.DATASET.NAME:
        case "vric":
            root = os.path.join(cfg.DATASET.PATH, "vric")
            dataset = VRIC(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform)
        case "veri":
            root = os.path.join(cfg.DATASET.PATH, "VeRi")
            dataset = VeRi(root=root, split=split, image_index=image_index, label_index=label_index, transform=transform)
        case _:
            raise ValueError("TODO: other datasets, or a bug in the code not parsing dataset through argparse.")

    return dataset

