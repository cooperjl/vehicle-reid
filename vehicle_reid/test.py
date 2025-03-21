import logging
import os

import torch

from vehicle_reid.config import cfg
from vehicle_reid.datasets import load_data
from vehicle_reid.model import init_model
from vehicle_reid.eval import eval_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def main():
    logger.info("Evaluating model...")

    dataset, dataloader = load_data("train")

    model = init_model(cfg.MODEL.ARCH, dataset.num_classes)
    model = model.to(device)

    eval_model(model)


if __name__ == "__main__":
    main()

