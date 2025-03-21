import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .base import VehicleReIdDataset


# TODO: update this to work again
class VRIC(VehicleReIdDataset):
    """VRIC Dataset

    :param img_dir (str): Root directory of the dataset.
    :param split (string, optional): The dataset splits, "train" (default), "val", or "test"
    :param transform (callable, optional): optional transform to be applied on a sample.
    :param target_transform (callable, optional): optional transform to be applied on a target.
    """

    def __init__(
            self, 
            root: str,
            split: str='train',
            index: Optional[str]=None,
            transform: Optional[Callable]=None,
    ) -> None:
        super().__init__(root, split, index, transform)
        
        self.name = "vric"
        self.name_col = 0
        self.id_col = 1

        match split:
            case 'train':
                self.img_dir = os.path.join(root, 'train_images')
                img_labels = os.path.join(root, 'vric_train.txt')
            case 'val':
                self.img_dir = os.path.join(root, 'probe_images')
                img_labels = os.path.join(root, 'vric_probe.txt')
            case 'test':
                self.img_dir = os.path.join(root, 'gallery_images')
                img_labels = os.path.join(root, 'vric_gallery.txt')
            case _:
                raise ValueError("split must be train, val, or test")

        self.img_labels = pd.read_csv(img_labels, sep=' ', header=None, usecols=[0, 1])

