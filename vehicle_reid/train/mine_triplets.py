import numpy as np
import numpy.ma as ma
import torch

from vehicle_reid import utils
from vehicle_reid.config import cfg

rng = None


def mine_triplets(
    dataset,
    gms_dict: dict,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triplet mining function, used to select triplets per batch.

    Credit: https://github.com/adhirajghosh/RPTM_reid/blob/183e1f77a0979ab2ffa08b0bdb1c43ef0f633ad5/train.py#L49

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
    global rng
    if rng is None:
        rng = np.random.default_rng(seed=cfg.MISC.SEED)

    triplets = torch.zeros(
        (cfg.SOLVER.BATCH_SIZE * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
        dtype=torch.float32,
    )
    tri_labels = torch.zeros((cfg.SOLVER.BATCH_SIZE * 3), dtype=torch.int64)

    for i, label in enumerate(labels):
        anchor = images[i]
        anchor_idx = targets[i].item()

        label_str = utils.pad_label(label.item(), dataset.name)
        index = int(indices[i].item())

        matches = gms_dict[label_str][index]

        # mean threshold, as suggested by RPTM paper
        threshold = np.mean(matches)

        # mask out values which are below the threshold, and select the smallest
        pos_idx = np.argmin(ma.masked_where(matches < threshold, matches))
        positive = dataset.get_by_index(label_str, pos_idx)
        pos_dic = dataset[positive]

        # select a random negative anchor
        neg_label = label_str
        while neg_label == label_str or neg_label not in dataset.label_index:
            neg_label = rng.choice(np.fromiter(dataset.label_index.keys(), dtype="<U4"))

        neg_idx = rng.integers(0, len(dataset.label_index[neg_label]))
        negative = dataset.get_by_index(neg_label, neg_idx)
        neg_dic = dataset[negative]

        triplets[i] = anchor
        triplets[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[0]
        triplets[i + (cfg.SOLVER.BATCH_SIZE * 2)] = neg_dic[0]

        tri_labels[i] = anchor_idx
        tri_labels[i + cfg.SOLVER.BATCH_SIZE] = pos_dic[4]
        tri_labels[i + (cfg.SOLVER.BATCH_SIZE * 2)] = neg_dic[4]

    return triplets, tri_labels
