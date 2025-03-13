from argparse import Namespace

import numpy as np
import torch
from tqdm import tqdm

from vehicle_reid import model
from vehicle_reid.datasets import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def eval_veri(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with veri metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
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

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    mAP = np.mean(all_AP)

    return all_cmc, mAP

@torch.no_grad()
def extract_features(model, dataloader, desc=""):
    x_features = []
    x_labels = []
    x_camids = []

    for images, labels, _, camids in tqdm(dataloader, desc=desc):
        images = images.float().to(device)
        features = model(images)
        features = features.to("cpu")
        x_features.append(features)
        x_labels.extend(labels)
        x_camids.extend(camids)

    x_features = torch.cat(x_features, 0)
    x_labels = np.asarray(x_labels)
    x_camids = np.asarray(x_camids)

    return x_features, x_labels, x_camids


@torch.no_grad()
def eval_model(model: model.ResNet, args: Namespace):
    model.eval()

    _, queryloader = load_data(args, "query")
    _, galleryloader = load_data(args, "gallery")

    q_features, q_labels, q_camids = extract_features(model, queryloader, desc="extracting features for query images")
    g_features, g_labels, g_camids = extract_features(model, galleryloader, desc="extracting features for gallery images")

    qn, gn = q_features.size(0), g_features.size(0)

    distmat = torch.pow(q_features, 2).sum(dim=1, keepdim=True).expand(qn, gn) + \
              torch.pow(g_features, 2).sum(dim=1, keepdim=True).expand(gn, qn).t()

    distmat.addmm_(q_features, g_features.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("computing evaluation metrics")

    cmc, mAP = eval_veri(distmat, q_labels, g_labels, q_camids, g_camids, 10)

    print("results:")
    print(f"mAP: {mAP:.2f}")

    for rank in [1, 5, 10]:
        print(f"Rank-{rank}: {cmc[rank - 1]:.2f}")

