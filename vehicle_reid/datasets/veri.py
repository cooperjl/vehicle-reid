import os
from typing import Callable, Optional

import pandas as pd

from .base import VehicleReIdDataset


class VeRi(VehicleReIdDataset):
    """
    VeRi-776 Dataset class.
    
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

        self.train_classes = 576
        self.name = "veri"
        self.name_col = "imageName"
        self.id_col = "vehicleID"
        self.cid_col = "cameraID"

        match split:
            case 'train':
                self.img_dir = os.path.join(root, 'image_train')
                img_labels = os.path.join(root, 'train_label.xml')
            case 'query':
                self.img_dir = os.path.join(root, 'image_query')
                img_labels = os.path.join(root, 'test_label.xml')
            case 'gallery':
                self.img_dir = os.path.join(root, 'image_test')
                img_labels = os.path.join(root, 'test_label.xml')
            case _:
                raise ValueError("split must be train, query, or gallery")

        self.img_labels = pd.read_xml(img_labels, xpath='.//Item', encoding='gb2312', parser="etree").iloc[:, 0:3]

        # filter train labels down to just queries if split is query
        if split == 'query':
            queries = pd.read_csv(os.path.join(root, 'name_query.txt'), header=None).squeeze()
            self.img_labels = self.img_labels[self.img_labels[self.name_col].isin(queries)].reset_index()

