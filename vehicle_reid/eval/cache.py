import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

@torch.no_grad()
def cached_features(model) -> dict:
    model.eval()
    
    if cfg.MODEL.CHECKPOINT:
        basename = os.path.split(os.path.basename(cfg.MODEL.CHECKPOINT))[0]
    else:
        basename = ""

    filepath = os.path.join(cfg.MISC.CACHE_PATH, f"{basename}-cache.pth")

    if os.path.isfile(filepath):
        cache_dict = torch.load(filepath, weights_only=False)
        logger.info("Loaded cached gallery features")
    else:
        _, galleryloader = load_data("gallery")

        features, labels, camids = extract_features(
            model, galleryloader, desc="extracting features for gallery images"
        )

        cache_dict = {
            "features": features,
            "labels": labels,
            "camids": camids,
        }

        if cfg.MODEL.CHECKPOINT:
            torch.save(cache_dict, filepath)
            logger.info("Cached gallery features")

    return cache_dict

@torch.no_grad()
def extract_features(model, dataloader, desc: str = ""):
    """
    Function which extracts the features over the dataset using the dataloader.

    Parameters
    ----------
    model : nn.Module
        Model to extract the features, expects forward to return a single tensor in evaluation mode.
    dataloader : DataLoader
        dataloader of the dataset to evaluate performance using.
    desc : str, optional
        Optional description label for tqdm progress bar.
    """
    x_features = []
    x_labels = []
    x_camids = []

    for images, labels, _, camids, _ in tqdm(dataloader, desc=desc, leave=False):
        images = images.float().to(model.device)
        features = model(images)
        if model.device != "cpu":
            features = features.to(torch.device("cpu"))

        x_features.append(features)
        x_labels.extend(labels)
        x_camids.extend(camids)

    x_features = torch.cat(x_features, 0)
    x_labels = np.asarray(x_labels)
    x_camids = np.asarray(x_camids)

    return x_features, x_labels, x_camids

