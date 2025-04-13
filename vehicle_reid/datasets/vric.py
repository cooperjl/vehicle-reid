import os
from typing import Callable, Optional

import pandas as pd

from .base import VehicleReIdDataset


class VRIC(VehicleReIdDataset):
    """
    VRIC Dataset class.

    Attributes 
    ----------
    train_classes : int
        Number of classes in the train split.
    name : str
        String representation of the dataset.
    name_col : str
        Label of the image name column in the dataframe.
    id_col : str
        Label of the id column in the dataframe.
    cid_col : str
        Label of the camera id column in the dataframe.
    img_dir : str
        Path to the image directory.
    img_labels : pd.DataFrame
        Main dataset dataframe containing images and labels.
    """

    def __init__(
        self, 
        root: str,
        split: str='train',
        image_index: Optional[str]=None,
        label_index: Optional[str]=None,
        transform: Optional[Callable]=None,
    ) -> None:
        """
        Initialise the class.

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
        super().__init__(root, split, image_index, label_index, transform)
        
        self.train_classes = 2811
        self.name = "vric"
        self.name_col = 0
        self.id_col = 1
        self.cid_col = 2

        match split:
            case 'train':
                self.img_dir = os.path.join(root, 'train_images')
                img_labels = os.path.join(root, 'vric_train.txt')
            case 'query':
                self.img_dir = os.path.join(root, 'probe_images')
                img_labels = os.path.join(root, 'vric_probe.txt')
            case 'gallery':
                self.img_dir = os.path.join(root, 'gallery_images')
                img_labels = os.path.join(root, 'vric_gallery.txt')
            case _:
                raise ValueError("split must be train, query, or gallery")

        self.img_labels = pd.read_csv(img_labels, sep=' ', header=None, usecols=[0, 1, 2])

