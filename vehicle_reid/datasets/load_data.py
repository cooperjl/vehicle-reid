from argparse import Namespace

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from vehicle_reid.datasets import match_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO have the args be global its stupid passing all this data around
def load_data(args: Namespace, split: str, batch_size: int = 16):
    dataset = match_dataset(args, split)

    # TODO: num workers 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return dataset, dataloader
