import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import numpy.ma as ma
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from vehicle_reid import gms, losses, utils
from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import eval_model
from vehicle_reid.model import init_model
from vehicle_reid.optimizer import init_optimizer
from vehicle_reid.utils import AverageMeter, load_checkpoint, save_checkpoint

rng = np.random.default_rng(seed=1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler = torch.GradScaler()

logger = logging.getLogger(__name__)

def train():
    """
    Main training function, to train a model using the parameters in the specified configuration file.
    """
    logger.info("Entering training loop...")

    triplet_loss_fn = losses.TripletLoss(margin=cfg.LOSS.MARGIN)
    ce_loss_fn = losses.CrossEntropyLoss(label_smoothing=0.1)

    gms_path = os.path.join(cfg.MISC.GMS_PATH, cfg.DATASET.NAME)
    gms_dict = gms.load_data(gms_path)

    dataset, dataloader = load_data("train")

    model = init_model(cfg.MODEL.ARCH, num_classes=dataset.train_classes, two_branch=cfg.MODEL.TWO_BRANCH, device=device)
    model = model.to(device)

    optimizer = init_optimizer(model.named_parameters())

    start_epoch = 0

    if cfg.MODEL.CHECKPOINT:
        start_epoch = load_checkpoint(cfg.MODEL.CHECKPOINT, model, optimizer)
    
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=cfg.SOLVER.GAMMA, last_epoch=start_epoch-1)

    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        logger.info(f"Epoch {epoch+1}:")
        train_one_epoch(model, optimizer, dataset, dataloader, gms_dict, triplet_loss_fn, ce_loss_fn, desc=f"epoch {epoch+1}")
        eval_model(model)
        scheduler.step()

        if (epoch+1) % cfg.MISC.SAVE_FREQ == 0:
            logger.info(f"Saving checkpoint at epoch {epoch+1}")
            save_checkpoint(epoch+1, model.state_dict(), optimizer.state_dict())


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    dataloader,
    gms_dict: dict, 
    triplet_loss_fn: losses.TripletLoss, 
    ce_loss_fn: losses.CrossEntropyLoss,
    desc: str="",
    ):
    """
    Singular epoch training function.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to train.
    optimizer : torch.optim.Optimizer
        Optimizer for use in training.
    dataset : datasets.VehicleReIdDataset
        Dataset instance being trained on.
    dataloader : DataLoader
        DataLoader configured for dataset.
    gms_dict : dict
        GMS feature match dictionary.
    triplet_loss_fn : losses.TripletLoss
        TripletLoss loss function instance.
    ce_loss_fn : losses.CrossEntropyLoss
        CrossEntropyLoss loss function instance.
    desc : str, optional
        Optional description label for tqdm progress bar.
    """
    losses = AverageMeter()
    times = AverageMeter()

    model.train()

    for batch_idx, (images, labels, indices, _, targets) in enumerate(tqdm(dataloader, desc=desc)):
        start = time.time()

        triplets, tri_labels = mine_triplets(dataset, gms_dict, images, labels, indices, targets)

        optimizer.zero_grad()

        triplets = triplets.to(device)
        tri_labels = tri_labels.to(device)

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
        times.update(time.time() - start)

        if batch_idx % 100 == 0:
            logger.info(f"Loss: {losses.val:.4f} ({losses.avg:.4f}), Time: {times.val:.3f}s ({times.avg:.3f}s)")


def mine_triplets(
        dataset,
        gms_dict: dict,
        images: torch.Tensor, 
        labels: torch.Tensor,
        indices: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triplet mining function, used to mine triplets per batch.

    Parameters
    ----------
    dataset : datasets.VehicleReIdDataset
        Dataset instance being trained on.
    gms_dict : dict
        GMS feature match dictionary.
    images : torch.Tensor
        Tensor containing a batch of anchor images with shape (batch_size, 3, image_height, image_width).
    labels : torch.Tensor
        Tensor containing a batch of anchor labels with shape (batch_size).
    indices : torch.Tensor
        Tensor containing relative index of anchors per identity with shape (batch_size).
    targets : torch.Tensor
        Tensor containing absolute index of anchors relative to the dataset with shape (batch_size).

    Returns
    -------
    triplets : torch.Tensor
        Tensor containing anchor, positive, and negative images with shape (batch_size * 3, 3, image_height, image_width).
    tri_labels : torch.Tensor
        Tensor containing labels of the anchors, positives (same as anchor),  
        and negatives (different to anchor) with shape (batch_size * 3).
    """
    triplets = torch.zeros((cfg.SOLVER.BATCH_SIZE * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), dtype=torch.float32)
    tri_labels = torch.zeros((cfg.SOLVER.BATCH_SIZE * 3), dtype=torch.int64)

    for i, label in enumerate(labels):
        anchor = images[i]
        anchor_idx = targets[i].item()

        label_str = utils.pad_label(label.item(), dataset.name)
        index = int(indices[i].item())

        matches = gms_dict[label_str][index]

        # mean threshold
        threshold = np.mean(matches)
        # max threshold
        # threshold = np.max(matches)

        match cfg.SOLVER.TRIPLET_SELECT:
            case "mean":
                threshold = np.mean(matches)
            case "mean-min":
                threshold = max(np.mean(matches), 50)
            case "min":
                threshold = 50
            case _:
                raise ValueError("Parameter TRIPLET_SELECT must be: \"mean\", \"mean-min\", or \"min\".")

        # mask out values which are below the threshold, and select the smallest
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

        triplets[i] = anchor
        triplets[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[0]
        triplets[i + (cfg.SOLVER.BATCH_SIZE* 2)] = neg_dic[0]
    
        tri_labels[i] = anchor_idx
        tri_labels[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[4]
        tri_labels[i + (cfg.SOLVER.BATCH_SIZE* 2)] = neg_dic[4]

    return triplets, tri_labels

