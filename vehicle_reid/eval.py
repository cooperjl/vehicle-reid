import logging

import numpy as np
import torch
from tqdm import tqdm

from vehicle_reid.datasets import load_data

logger = logging.getLogger(__name__)

@torch.no_grad()
def eval_model(model):
    """
    Main evaluation function, to evaluate a model using the parameters in the specified configuration file,
    such as the dataset.

    Parameters
    ----------
    model : nn.Module
        Model to extract the features, expects forward to return a single tensor in evaluation mode.
    """
    _, queryloader = load_data("query")
    _, galleryloader = load_data("gallery")

    distmat, q_labels, g_labels, q_camids, g_camids = calculate_distmat(model, queryloader, galleryloader)
    cmc, mAP = eval_veri(distmat, q_labels, g_labels, q_camids, g_camids, 10)

    logger.info(f"mAP: {mAP:.1%}")

    for rank in [1, 5, 10]:
        logger.info(f"Rank-{rank}: {cmc[rank - 1]:.1%}")


@torch.no_grad()
def calculate_distmat(model, queryloader, galleryloader):
    """
    Function which calculates the distmat using the parameters in the specified configuration file, such as the dataset.
    It also returns important parameters for the evaluation of the model, such as labels and camera ids.

    Parameters
    ----------
    model : nn.Module
        Model to extract the features, expects forward to return a single tensor in evaluation mode.
    queryloader : DataLoader
        Dataloader for the query set.
    galleryloader : DataLoader
        Dataloader for the gallery set.

    Returns
    -------
    distmat : np.ndarray
        distance matrix of shape (num_query, num_gallery).
    q_labels : np.ndarray
        1d array containing identities of each query instance.
    g_labels : np.ndarray
        1d array containing identities of each gallery instance.
    q_camids : np.ndarray
        1d array containing camera ids under which each query instance is captured.
    g_camids : np.ndarray
        1d array containing camera ids under which each gallery instance is captured.
    """
    model.eval()

    q_features, q_labels, q_camids = extract_features(model, queryloader, desc="extracting features for query images")
    g_features, g_labels, g_camids = extract_features(model, galleryloader, desc="extracting features for gallery images")

    qn, gn = q_features.size(0), g_features.size(0)

    distmat = torch.pow(q_features, 2).sum(dim=1, keepdim=True).expand(qn, gn) + \
              torch.pow(g_features, 2).sum(dim=1, keepdim=True).expand(gn, qn).t()

    distmat.addmm_(q_features, g_features.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    return distmat, q_labels, g_labels, q_camids, g_camids


@torch.no_grad()
def extract_features(model, dataloader, desc: str=""):
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


@torch.no_grad()
def eval_veri(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray, q_camids: np.ndarray, g_camids: np.ndarray, max_rank: int=10):
    """
    Evaluation with veri metric: for each query identity, its gallery images from the same camera view are discarded.
    Credit: https://github.com/KaiyangZhou/deep-person-reid/blob/566a56a2cb255f59ba75aa817032621784df546a/torchreid/metrics/rank.py#L94

    Parameters
    ----------
    distmat : np.ndarray
        distance matrix of shape (num_query, num_gallery).
    q_pids : np.ndarray
        1d array containing identities of each query instance.
    g_pids : np.ndarray
        1d array containing identities of each gallery instance.
    q_camids : np.ndarray
        1d array containing camera ids under which each query instance is captured.
    g_camids : np.ndarray
        1d array containing camera ids under which each gallery instance is captured.
    max_rank : int, optional
        maximum CMC rank to be computed (default is 10).

    Returns
    -------
    all_cmc : np.ndarray
        1d array of length max_rank, containing rank 1-max_rank cmc results.
    mAP : np.float32
        mean average precision.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        logger.warning(f"Number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in tqdm(range(num_q), desc="computing evaluation metrics", leave=False):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if not num_valid_q > 0:
        error = "Not all query identities appear in gallery"
        logger.error(error)
        raise AssertionError(error)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)

    return all_cmc, mAP

