import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from vehicle_reid.datasets.base import VehicleReIdDataset


class VeRi(VehicleReIdDataset):
    """VeRi Dataset

    :param img_dir (str): Root directory of the dataset.
    :param split (string, optional): The dataset splits, "train" (default), "val", or "test"
    :param transform (callable, optional): optional transform to be applied on a sample.
    :param target_transform (callable, optional): optional transform to be applied on a target.
    """

    def __init__(
            self, 
            root: str,
            split: str='train',
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None
    ) -> None:
        super().__init__(root, split, transform, target_transform)

        match split:
            case 'train':
                self.img_dir = os.path.join(root, 'image_train')
                img_labels = os.path.join(root, 'vric_train.txt')
            case 'val':
                self.img_dir = os.path.join(root, 'image_query')
                img_labels = os.path.join(root, 'vric_probe.txt')
            case 'test':
                self.img_dir = os.path.join(root, 'image_test')
                img_labels = os.path.join(root, 'vric_gallery.txt')
            case _:
                raise ValueError("split must be train, val, or test")

        # discard the camera information with usecols, as it is not useful for this project
        self.img_labels = pd.read_csv(img_labels, sep=' ', header=None, usecols=[0, 1])

