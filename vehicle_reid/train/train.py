import logging
import os
import time

import torch
from tqdm import tqdm

from vehicle_reid import utils
from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import eval_model
from vehicle_reid.model import init_model

from .losses import CrossEntropyLoss, TripletLoss
from .mine_triplets import mine_triplets
from .optimizer import init_optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scaler = torch.GradScaler()

logger = logging.getLogger(__name__)


def train_model():
    """
    Main training function, to train a model using the parameters in the specified configuration file.
    """
    logger.info("Entering training loop...")

    triplet_loss_fn = TripletLoss(margin=cfg.LOSS.MARGIN)
    ce_loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    rel_path = os.path.join(cfg.DATASET.PATH, cfg.DATASET.REL_PATH, cfg.DATASET.NAME)
    rel_dict = utils.load_relational_data(rel_path)

    dataset, dataloader = load_data("train")

    model = init_model(
        cfg.MODEL.ARCH,
        num_classes=dataset.train_classes,
        two_branch=cfg.MODEL.TWO_BRANCH,
        device=device,
    )
    model = model.to(device)

    optimizer = init_optimizer(model.named_parameters())

    start_epoch = 0

    if cfg.MODEL.CHECKPOINT:
        start_epoch = utils.load_checkpoint(cfg.MODEL.CHECKPOINT, model, optimizer)

    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        logger.info(f"Epoch {epoch + 1}:")
        train_one_epoch(
            model,
            optimizer,
            dataset,
            dataloader,
            rel_dict,
            triplet_loss_fn,
            ce_loss_fn,
            desc=f"epoch {epoch + 1}",
        )
        eval_model(model)

        if (epoch + 1) % cfg.MISC.SAVE_FREQ == 0:
            logger.info(f"Saving checkpoint at epoch {epoch + 1}")
            utils.save_checkpoint(epoch + 1, model.state_dict(), optimizer.state_dict())


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    dataloader,
    rel_dict: dict,
    triplet_loss_fn: TripletLoss,
    ce_loss_fn: CrossEntropyLoss,
    desc: str = "",
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
    rel_dict : dict
        Relational data dictionary.
    triplet_loss_fn : TripletLoss
        TripletLoss loss function instance.
    ce_loss_fn : CrossEntropyLoss
        CrossEntropyLoss loss function instance.
    desc : str, optional
        Optional description label for tqdm progress bar.
    """
    losses = utils.AverageMeter()
    times = utils.AverageMeter()

    model.train()

    for batch_idx, (images, labels, indices, _, targets) in enumerate(
        tqdm(dataloader, desc=desc)
    ):
        start = time.time()

        triplets, tri_labels = mine_triplets(
            dataset, rel_dict, images, labels, indices, targets
        )

        optimizer.zero_grad()

        triplets = triplets.to(device)
        tri_labels = tri_labels.to(device)

        outputs, features = model(triplets)

        ce_loss = ce_loss_fn(
            outputs[0 : cfg.SOLVER.BATCH_SIZE],
            tri_labels[0 : cfg.SOLVER.BATCH_SIZE],
        )
        triplet_loss = triplet_loss_fn(features, tri_labels)

        loss = (cfg.LOSS.LAMBDA_TRI * triplet_loss) + (cfg.LOSS.LAMBDA_CE * ce_loss)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), tri_labels.size(0))
        times.update(time.time() - start)

        if batch_idx % 100 == 0:
            logger.info(
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}), Time: {times.val:.3f}s ({times.avg:.3f}s)"
            )
