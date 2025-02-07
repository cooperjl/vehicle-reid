import os
from collections import defaultdict
from typing import Callable, List, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
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
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None
    ) -> None:
        super().__init__(root, split, transform, target_transform)

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

        # discard the camera information with usecols, as it is not useful for this project
        self.img_labels = pd.read_csv(img_labels, sep=' ', header=None, usecols=[0, 1])


if __name__ == "__main__":
    """Basic visualisation for the dataset"""

    vric_dataset = VRIC(root='data/vric/', split='train')

    fig = plt.figure()

    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.tight_layout()
        ax.axis('off')

        sample_idx = torch.randint(len(vric_dataset), size=(1,)).item()
        img, label = vric_dataset[sample_idx]
        plt.title(label)
        plt.imshow(img)

    plt.show()

