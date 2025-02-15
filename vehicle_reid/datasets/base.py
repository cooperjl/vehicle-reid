import json
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


class VehicleReIdDataset(Dataset):
    """Vehicle ReId dataset base class.

    Need to define self.img_dir and self.img_labels in init function when used, as seen in classes in this file.
    """

    def __init__(
        self, 
        root: str,
        split: str='train',
        image_index: Optional[str]=None,
        label_index: Optional[str]=None,
        transform: Optional[Callable]=None,
        target_transform: Optional[Callable]=None,
    ) -> None:
        self.root = root
        self.split = split
        
        if image_index:
            with open(image_index, 'r') as f:
                self.image_index = json.load(f)
        if label_index:
            with open(label_index, 'r') as f:
                self.label_index = json.load(f)
                self.max_label = max(self.label_index.keys())

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.image_index is None:
            index = None
        elif self.img_labels.iloc[idx, 0] not in self.image_index:
            index = 0
        else:
            index = self.image_index[self.img_labels.iloc[idx, 0]][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, index

    def get_grouped(self):
        # group by classes
        img_path_with_labels = self.img_labels.copy()
        img_path_with_labels.iloc[:, 0] = self.img_dir + os.sep + img_path_with_labels.iloc[:, 0]

        grouped = img_path_with_labels.groupby(self.id_col)[self.name_col]

        return grouped

    def get_random_label(self):
        label = self.img_labels.sample(axis=0)[1].item()
        return self.img_labels.loc[self.img_labels[1] == label][0].reset_index()

    def get_by_index(self, label: str, index: int) -> torch.Tensor:
        """Get by the label index."""
        image_name = self.label_index[label][index]
        img_path = os.path.join(self.img_dir, image_name)
        image = decode_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

