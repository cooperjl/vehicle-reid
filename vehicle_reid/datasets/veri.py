import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from vehicle_reid.datasets.base import VehicleReIdDataset


class VeRi(VehicleReIdDataset):
    """VeRi Dataset

    :param img_dir (str): Root directory of the dataset.
    :param split (string, optional): The dataset splits, "train" (default), "query", or "gallery" (test)
    :param transform (callable, optional): optional transform to be applied on a sample.
    :param target_transform (callable, optional): optional transform to be applied on a target.
    """

    def __init__(
            self, 
            root: str,
            split: str='train',
            image_index: Optional[str]=None,
            label_index: Optional[str]=None,
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None
    ) -> None:
        super().__init__(root, split, image_index, label_index, transform, target_transform)

        self.name = "veri"
        self.name_col = "imageName"
        self.id_col = "vehicleID"

        match split:
            case 'train':
                self.img_dir = os.path.join(root, 'image_train')
                img_labels = os.path.join(root, 'train_label.xml')
            #case 'val':
            #    self.img_dir = os.path.join(root, 'image_query')
            #    img_labels = os.path.join(root, 'vric_probe.txt')
            case 'gallery':
                self.img_dir = os.path.join(root, 'image_test')
                img_labels = os.path.join(root, 'test_label.xml')
            case _:
                raise ValueError("split must be train, query, or gallery")

        self.img_labels = pd.read_xml(img_labels, xpath='.//Item', encoding='gb2312', parser="etree").iloc[:, 0:2]

