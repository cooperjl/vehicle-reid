import os
from collections import defaultdict
from typing import Callable, List, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset


class VRIC(Dataset):
    """VRIC Dataset

    Args:
        img_dir (str): Root directory of the dataset.
        split (string, optional): The dataset splits, "train" (default), "val", or "test"
        transform (callable, optional): optional transform to be applied on a sample.
        target_transform (callable, optional): optional transform to be applied on a target.
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

        self.img_labels = pd.read_csv(img_labels, sep=' ', header=None)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_dict(self): # -> dict[str, list[str]]:
        class_images = defaultdict(list)
        print(self.img_labels[0].groupby(self.img_labels[1]).head())

    def get_random_label(self):
        label = self.img_labels.sample(axis=0)[1].item()
        return self.img_labels.loc[self.img_labels[1] == label][0].reset_index()




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

