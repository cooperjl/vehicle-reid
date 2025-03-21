import logging
import os
from contextlib import nullcontext

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from tqdm import tqdm

from vehicle_reid import gms, losses, utils
from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import eval_model
from vehicle_reid.model import init_model
from vehicle_reid.optimizer import init_optimizer
from vehicle_reid.utils import AverageMeter

#torch.multiprocessing.set_sharing_strategy('file_system')

rng = np.random.default_rng(seed=1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler = torch.GradScaler()

logger = logging.getLogger(__name__)

def train_one_epoch(
    model,
    optimizer,
    dataset,
    dataloader,
    gms_dict: dict, 
    triplet_loss_fn: losses.TripletLoss, 
    ce_loss_fn: losses.CrossEntropyLoss,
    desc=""
    ):
    """
    Singular epoch training function.
    """
    losses = AverageMeter()

    model.train()

    for p in model.parameters():
        p.requires_grad = True  # open all layers

    for batch_idx, (images, labels, indices, _, idxs) in enumerate(tqdm(dataloader, desc=desc)):
        triplets, tri_labels = mine_triplets(dataset, gms_dict, images, labels, indices, idxs)

        optimizer.zero_grad()

        triplets = triplets.to(device)
        tri_labels = tri_labels.to(device)

        # global module TODO rename and reorder
        with torch.autocast(device_type=cfg.MODEL.DEVICE) if cfg.SOLVER.AMP else nullcontext():
            outputs, features = model(triplets)

            # need to decide whether to calculate cross entropy loss for whole triplet or just anchor
            ce_loss = ce_loss_fn(outputs[0:cfg.SOLVER.BATCH_SIZE], tri_labels[0:cfg.SOLVER.BATCH_SIZE])
            triplet_loss = triplet_loss_fn(features, tri_labels)

            loss = (cfg.LOSS.LAMBDA_TRI * triplet_loss) + (cfg.LOSS.LAMBDA_CE * ce_loss)
        
        if cfg.SOLVER.AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), tri_labels.size(0)) # TODO: amp support

        if batch_idx % 100 == 0:
            logger.info(f"Loss: {losses.val:.4f} ({losses.avg:.4f})") # TODO: log epoch info as well


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
    plt.imshow(pmaps[cfg.SOLVER.BATCH_SIZE])
    fig.add_subplot(2, 3, 3)
    plt.imshow(pmaps[cfg.SOLVER.BATCH_SIZE*2])
    fig.add_subplot(2, 3, 4)
    plt.imshow(triplets[0].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 5)
    plt.imshow(triplets[cfg.SOLVER.BATCH_SIZE].to(torch.uint8).permute(1, 2, 0))
    fig.add_subplot(2, 3, 6)
    plt.imshow(triplets[cfg.SOLVER.BATCH_SIZE*2].to(torch.uint8).permute(1, 2, 0))


    plt.show()


def mine_triplets(dataset, gms_dict: dict, images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, idxs: torch.Tensor):
    triplets = torch.zeros((cfg.SOLVER.BATCH_SIZE * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), dtype=torch.float32)
    tri_labels = torch.zeros((cfg.SOLVER.BATCH_SIZE * 3), dtype=torch.int64)

    for i, label in enumerate(labels):
        anchor = images[i]
        anchor_idx = idxs[i].item()

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
        pos_dic = dataset[positive]

        # select a random negative anchor
        neg_label = label_str
        while neg_label == label_str or neg_label not in dataset.label_index:
            neg_label = rng.choice(np.fromiter(dataset.label_index.keys(), dtype='<U4'))

        neg_idx = rng.integers(0, len(dataset.label_index[neg_label]))
        negative = dataset.get_by_index(neg_label, neg_idx)
        neg_dic = dataset[negative]

        if anchor_idx != pos_dic[4]:
            logger.error("Anchor and positive are not the same class, triplet mining has gone wrong")

        if anchor_idx == neg_dic[4]:
            logger.error("Anchor and negative are the same class, triplet mining has gone wrong")

        triplets[i] = anchor
        triplets[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[0]
        triplets[i + (cfg.SOLVER.BATCH_SIZE* 2)] = neg_dic[0]
    
        tri_labels[i] = anchor_idx
        tri_labels[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[4]
        tri_labels[i + (cfg.SOLVER.BATCH_SIZE* 2)] = neg_dic[4]

    return triplets, tri_labels


def train():
    logger.info("Entering training loop...")

    triplet_loss_fn = losses.TripletLoss(margin=1.0)
    ce_loss_fn = losses.CrossEntropyLoss(label_smoothing=0.1)

    gms_path = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    gms_dict = gms.load_data(gms_path)

    dataset, dataloader = load_data("train") # TODO: integrate gms_dict into load_data

    model = init_model(cfg.MODEL.ARCH, 576)
    model = model.to(device)
    optimizer = init_optimizer(model.parameters())


    for epoch in range(cfg.SOLVER.EPOCHS):
        train_one_epoch(model, optimizer, dataset, dataloader, gms_dict, triplet_loss_fn, ce_loss_fn, desc=f"epoch {epoch+1}")
        eval_model(model)

