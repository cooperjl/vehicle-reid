import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import cv2 as cv

from vehicle_reid import args, gms, utils
from vehicle_reid.datasets import load_data
from vehicle_reid.model import Model

CFG = Namespace()
rng = np.random.default_rng()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = args.add_subparser(name="train", help="train the model")
    parser.add_argument('dataset', metavar='dataset',
                        choices=args.DATASETS,
                        help='the name of the dataset to train on')
    parser.add_argument('--data-path', type=args.dir_path,
                        default=args.DEFAULTS.data_path,
                        help='path where the datasets are stored (default: %(default)s)')
    parser.add_argument('--gms-path', type=args.dir_path,
                        default=args.DEFAULTS.gms_path,
                        help='path to load gms matches from (default: %(default)s)')
    parser.add_argument('--epochs', type=int,
                        required=True,
                        help='number of epochs for training')
    parser.add_argument('--width', type=int,
                        default=args.DEFAULTS.width,
                        help='width to resize images to (default: %(default)s)')
    parser.add_argument('--height', type=int,
                        default=args.DEFAULTS.height,
                        help='width to resize images to (default: %(default)s)')
    parser.set_defaults(func=train)


def train_one_epoch(model: Model, dataset, dataloader, gms_dict: dict):
    """
    Singular epoch training function. TODO
    """

    model.train()

    #trainX = torch.zeros((batch_size * 3, 3, height, width), dtype=torch.float32)
    #trainY = torch.zeros((batch_size * 3), dtype=torch.float32)


    for images, labels, indices in dataloader:
        inputs = mine_triplets(dataset, gms_dict, images, labels, indices)
        inputs = inputs.to(device)

        # optimizer.zero_grad()
        outputs = Model(inputs)
        #loss = 

        break

def mine_triplets(dataset, gms_dict: dict, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor):
    batch_size = 16 # TODO: CFG.batch_size
    triplets = torch.zeros((batch_size * 3, 3, CFG.height, CFG.width), dtype=torch.float32)

    for i, label in enumerate(labels):
        label = utils.pad_label(label.item(), dataset.name)
        index = int(indices[i].item())

        matches = gms_dict[label][index]

        # mean threshold
        threshold = np.mean(matches)
        # max threshold
        # threshold = np.max(matches)

        # mask out values which are below the threshold, and select the smallest
        # maybe if below 50, we switch to max instead of mean? NOTE: important to remember this
        pos_idx = np.argmin(ma.masked_where(matches < threshold, matches))
        pos_anchor = dataset.get_by_index(label, pos_idx)

        # select a random negative anchor
        neg_label = label
        while neg_label == label or neg_label not in dataset.label_index:
            neg_label = rng.choice(np.fromiter(dataset.label_index.keys(), dtype='<U4'))

        neg_idx = rng.integers(0, len(dataset.label_index[neg_label]))
        neg_anchor = dataset.get_by_index(neg_label, neg_idx)

        triplets[i] = images[i]
        triplets[i + batch_size] = pos_anchor
        triplets[i + (batch_size * 2)] = neg_anchor

    return triplets


def train(args: Namespace):
    print("Train called")

    global CFG 
    CFG = args # TODO have a global args

    gms_path = os.path.join(args.gms_path, args.dataset)
    gms_dict = gms.load_data(gms_path)

    model = Model()
    dataset, dataloader = load_data(args, "train")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}:")

        train_one_epoch(model, dataset, dataloader, gms_dict)

