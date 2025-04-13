import json
import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


class VehicleReIdDataset(Dataset):
    """
    Vehicle re-identification dataset base class.

    Need to define self.img_dir and self.img_labels in init function when used, as seen in classes in this module.
    """

    def __init__(
        self, 
        root: str,
        split: str="train",
        image_index: Optional[str]=None,
        label_index: Optional[str]=None,
        transform: Optional[Callable]=None,
    ) -> None:
        """
        Parameters
        ----------
        root : str
            root directory containing the datasets.
        split : str, optional
            which split of the dataset to use. Default value is "train".
        image_index : str, optional
            path to the image index file generated at the same time as the gms data. Optional, since to generate these files
            that script needs to access the dataset.
        label_index : str, optional
            path to the label index file generated at the same time as the gms data. Optional, since to generate these files
            that script needs to access the dataset.
        transform : callable, optional
            transform to apply to the images.
        """
        self.root = root
        self.split = split
        
        if image_index:
            with open(image_index, 'r') as f:
                self.image_index = json.load(f)
        if label_index:
            with open(label_index, 'r') as f:
                self.label_index = json.load(f)
                self.max_label = max(self.label_index.keys())
                self.label_index_keys = [int(k) for k in self.label_index.keys()]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, self.name_col])
        image = decode_image(img_path)
        label = self.img_labels.loc[idx, self.id_col]
        cam_id = self.img_labels.loc[idx, self.cid_col]
        target = self.label_index_keys.index(label) if label in self.label_index_keys else torch.empty(0)
        if self.image_index is None:
            index = None
        elif self.img_labels.loc[idx, self.name_col] not in self.image_index:
            index = 0
        else:
            index = self.image_index[self.img_labels.loc[idx, self.name_col]][1]
        if self.transform:
            image = self.transform(image)

        return image, label, index, cam_id, target

    def get_grouped(self):
        """Used to get the dataset grouped by classes, used for the gms script."""
        img_path_with_labels = self.img_labels.copy()
        img_path_with_labels.iloc[:, 0] = self.img_dir + os.sep + img_path_with_labels.iloc[:, 0]

        grouped = img_path_with_labels.groupby(self.id_col)[self.name_col]

        return grouped

    def get_random_label(self):
        """Get a random label, used for purposes such as visualisation."""
        # TODO: consider the difference between this and the version in train.py
        label = self.img_labels.sample(axis=0)[1].item()
        return self.img_labels.loc[self.img_labels[1] == label][0].reset_index()

    def get_by_index(self, label: str, index: int) -> int:
        """Get by the label index."""
        return self.label_index[label][index]
    
