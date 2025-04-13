import os

from torch.utils.data import DataLoader

from vehicle_reid.config import cfg
from vehicle_reid.datasets import VRIC, VeRi
from vehicle_reid.datasets.transforms import transform_test, transform_train


def load_data(split: str):
    """
    Loads the dataset and the dataloader.

    Parameters
    ----------
    split : str
        The dataset split to use.
    
    Returns
    -------
    dataset : dataset.VehicleReIdDataset
        The dataset loaded using the configuration file.
    dataloader : DataLoader
        The dataloader which loads the dataset.
    """
    batch_size = cfg.SOLVER.BATCH_SIZE if split == "train" else cfg.TEST.BATCH_SIZE
    
    dataset = match_dataset(split)

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    return dataset, dataloader

def match_dataset(split: str):
    """
    Matches the dataset split and name from the configuration file, and creates the dataset objects.

    Parameters
    ----------
    split : str
        The dataset split to use.

    Returns
    -------
    dataset : dataset.VehicleReIdDataset
        The dataset loaded using the configuration file.
    """
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

