import torch
import torchvision.transforms.v2 as transforms

from vehicle_reid.config import cfg


def transform_train():
    transform = [
        transforms.Resize(size=cfg.INPUT.WIDTH, antialias=True),
        transforms.CenterCrop(size=(cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT)),
    ]

    if cfg.INPUT.AUGMENT:
        transform.extend(
            [
                transforms.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.15, saturation=0, hue=0
                ),
            ]
        )

    transform.extend(
        [
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD),
        ]
    )

    if cfg.INPUT.RAND_ERASE:
        transform.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        )

    return transforms.Compose(transform)


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
