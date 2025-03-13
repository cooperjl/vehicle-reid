import os
from argparse import BooleanOptionalAction, Namespace
from contextlib import nullcontext

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

from vehicle_reid import gms, losses, utils
from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import eval_model
from vehicle_reid.model import cresnet50
from vehicle_reid.optimizer import init_optimizer

torch.multiprocessing.set_sharing_strategy('file_system')

triplet_loss_fn = losses.TripletLoss()
ce_loss_fn = losses.CrossEntropyLoss()

rng = np.random.default_rng()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = cfg.SOLVER.BATCH_SIZE

scaler = torch.GradScaler()


def train_one_epoch(model, optimizer, dataset, dataloader, gms_dict: dict, desc=""):
    """
    Singular epoch training function.
    """

    model.train()

    for images, labels, indices, _ in tqdm(dataloader, desc=desc):
        triplets, tri_labels = mine_triplets(dataset, gms_dict, images, labels, indices)

        triplets = triplets.to(device)
        tri_labels = tri_labels.to(device)

        optimizer.zero_grad()
        # global module TODO rename and reorder
        with torch.autocast(device_type=cfg.MODEL.DEVICE) if cfg.SOLVER.AMP else nullcontext():
            outputs, features = model(triplets)

            # need to decide whether to calculate cross entropy loss for whole triplet or just anchor
            ce_loss = ce_loss_fn(outputs[0:batch_size], tri_labels[0:batch_size])
            
            #magic_gcn_work_time_todo(features, tribak)

            triplet_loss = triplet_loss_fn(features, tri_labels)

            loss = (1.0 * triplet_loss) + (1.0 * ce_loss)
        
        if cfg.SOLVER.AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def magic_gcn_work_time_todo(featuremaps: torch.Tensor, triplets: torch.Tensor):
    # TODO This should not be here! this is really a test okayyyy
    size = featuremaps.size()
    pmaps = []
    for map in featuremaps:
        map = map.squeeze(0)
        pmaps.append((torch.sum(map, 0)/map.shape[0]).to('cpu').detach().numpy())
    #(top_left, top_right, bottom_left, bottom_right, centre) = transforms.FiveCrop(size=(150, 150))(featuremaps)
    fig = plt.figure(figsize=(2, 3))
    fig.add_subplot(2, 3, 1)
    plt.imshow(pmaps[0])
    fig.add_subplot(2, 3, 2)
    plt.imshow(pmaps[batch_size])
    fig.add_subplot(2, 3, 3)
    plt.imshow(pmaps[batch_size*2])
    fig.add_subplot(2, 3, 4)
    plt.imshow(triplets[0].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 5)
    plt.imshow(triplets[batch_size].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 6)
    plt.imshow(triplets[batch_size*2].to(torch.uint8).permute(1, 2, 0))


    plt.show()


def mine_triplets(dataset, gms_dict: dict, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor):
    triplets = torch.zeros((batch_size * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), dtype=torch.float32)
    tri_labels = torch.zeros((batch_size * 3), dtype=torch.int64)

    for i, label in enumerate(labels):
        anchor = images[i]

        label_str = utils.pad_label(label.item(), dataset.name)
        index = int(indices[i].item())

        matches = gms_dict[label_str][index]

        # mean threshold
        threshold = np.mean(matches)
        # max threshold
        # threshold = np.max(matches)

        # mask out values which are below the threshold, and select the smallest
        # maybe if below say 50, we switch to max instead of mean? NOTE: important to remember this
        pos_idx = np.argmin(ma.masked_where(matches < threshold, matches))
        positive = dataset.get_by_index(label_str, pos_idx)

        # select a random negative anchor
        neg_label = label_str
        while neg_label == label_str or neg_label not in dataset.label_index:
            neg_label = rng.choice(np.fromiter(dataset.label_index.keys(), dtype='<U4'))

        neg_idx = rng.integers(0, len(dataset.label_index[neg_label]))
        negative = dataset.get_by_index(neg_label, neg_idx)

        triplets[i] = anchor
        triplets[i + batch_size] = positive
        triplets[i + (batch_size * 2)] = negative

        tri_labels[i] = label
        tri_labels[i + batch_size] = label
        tri_labels[i + (batch_size * 2)] = int(neg_label)

    return triplets, tri_labels


def train():
    print("Train called")

    gms_path = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    gms_dict = gms.load_data(gms_path)

    dataset, dataloader = load_data("train")

    model = cresnet50(dataset.num_classes)
    model = model.to(device)
    optimizer = init_optimizer("sgd", model.parameters())

    for epoch in range(cfg.SOLVER.EPOCHS):
        train_one_epoch(model, optimizer, dataset, dataloader, gms_dict, desc=f"epoch {epoch+1}")
        eval_model(model)

