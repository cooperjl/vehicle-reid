import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.eval import calculate_distmat
from vehicle_reid.model import init_model
from vehicle_reid.utils import load_checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def configure_ax(ax, colour):
    ax.patch.set_edgecolor(colour)
    ax.patch.set_linewidth(8.0)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


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
    q_idx =  np.random.randint(len(queryset))

    # Query image
    q_img, q_label, _, q_camid, _ = queryset[q_idx]
    q_img = q_img.permute(1, 2, 0)
    # Query plot
    ax = plt.subplot(1, 11, 1)
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
            ax = plt.subplot(1, 11, rank_idx+1)
            colour = "limegreen" if q_label == g_label else "red"
            configure_ax(ax, colour)
            plt.title(g_label)
            plt.imshow(g_img)

            rank_idx += 1
            if rank_idx > 10:
                break

    plt.show()

def visualise_dataset():
    """
    Main visualise function, which loads the model and calls the function to visualise the ranked results.
    """
    # use images from the train set
    queryset, queryloader = load_data("query", normalise=False)
    galleryset, galleryloader = load_data("gallery", normalise=False)

    model = init_model(cfg.MODEL.ARCH, two_branch=cfg.MODEL.TWO_BRANCH, device=device)
    model = model.to(device)
    model.eval()

    if cfg.MODEL.CHECKPOINT:
        load_checkpoint(cfg.MODEL.CHECKPOINT, model)

    distmat, *_ = calculate_distmat(model, queryloader, galleryloader)
    visualise_ranked_results(distmat, queryset, galleryset)

