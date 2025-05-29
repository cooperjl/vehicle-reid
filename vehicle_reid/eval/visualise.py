import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data, match_dataset
from vehicle_reid.model import init_model
from vehicle_reid.utils import load_checkpoint

from .eval import calculate_distmat
from .reranking import reranking

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def visualise():
    """
    Main visualise function, which loads the model and calls the function to visualise the ranked results.
    """
    # need normalised queries to actually search with
    _, queryloader = load_data("query", normalise=True)

    # need unnormalised datasets to get clean images
    clean_queryset = match_dataset("query", normalise=False)
    clean_galleryset = match_dataset("gallery", normalise=False)

    model = init_model(cfg.MODEL.ARCH, two_branch=cfg.MODEL.TWO_BRANCH, device=device)
    model = model.to(device)
    model.eval()

    if cfg.MODEL.CHECKPOINT:
        load_checkpoint(cfg.MODEL.CHECKPOINT, model)

    distmat, q_features, g_features, *_ = calculate_distmat(model, queryloader)

    if cfg.DATASET.NAME == "veri":
        distmat = reranking(q_features, g_features, k1=40, k2=9, lambda_value=0.3)

    # Continuously generate outputs until terminated by the user with Ctrl+C
    while True:
        visualise_ranked_results(distmat, clean_queryset, clean_galleryset)


def visualise_ranked_results(distmat, queryset, galleryset):
    """
    Visualise top 10 results on a random query image. Query image are outlined in black, and positive and negative results
    in green and red respectively.

    Parameters
    ----------
    distmat : np.ndarray
        Distance matrix of shape (num_query, num_gallery).
    queryset : datasets.VehicleReIdDataset
        Dataset of the query images.
    galleryset : datasets.VehicleReIdDataset
        Dataset of the gallery images.
    """
    plt.tight_layout()

    indices = np.argsort(distmat, axis=1)
    q_idx = np.random.randint(indices.shape[0])

    # Query image
    q_img, q_label, _, q_camid, _ = queryset[q_idx]
    q_img = q_img.permute(1, 2, 0)
    # Query plot
    ax = plt.subplot(2, 5, 1)
    configure_ax(ax, "black")
    plt.title(q_label)
    plt.imshow(q_img)

    rank_idx = 1
    for g_idx in indices[q_idx, :]:
        # Gallery image
        g_img, g_label, _, g_camid, _ = galleryset[g_idx]
        g_img = g_img.permute(1, 2, 0)

        invalid = (q_label == g_label) and (q_camid == g_camid)
        if not invalid:
            # Gallery plot
            ax = plt.subplot(2, 5, rank_idx + 1)
            colour = "limegreen" if q_label == g_label else "red"
            configure_ax(ax, colour)
            plt.title(g_label)
            plt.imshow(g_img)

            rank_idx += 1
            if rank_idx >= 10:
                break

    plt.subplots_adjust(
        left=0.02, bottom=0.06, right=0.98, top=0.94, wspace=0.2, hspace=0.1
    )
    plt.show()


def configure_ax(ax, colour):
    ax.patch.set_edgecolor(colour)
    ax.patch.set_linewidth(8.0)
    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
