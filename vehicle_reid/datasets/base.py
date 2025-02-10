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
        transform: Optional[Callable]=None,
        target_transform: Optional[Callable]=None
    ) -> None:
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        #image = cv.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_grouped(self):
        # group by classes
        img_path_with_labels = self.img_labels.copy()
        img_path_with_labels[0] = self.img_dir + os.sep + img_path_with_labels[0]

        grouped = img_path_with_labels.groupby(1)[0]

        return grouped

    def get_random_label(self):
        label = self.img_labels.sample(axis=0)[1].item()
        return self.img_labels.loc[self.img_labels[1] == label][0].reset_index()

