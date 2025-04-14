import logging

import torch

from vehicle_reid.config import cfg
from vehicle_reid.datasets import match_dataset
from vehicle_reid.eval import eval_model
from vehicle_reid.model import init_model
from vehicle_reid.utils import load_checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def main():
    """
    Main testing function, which initialises the model, loads a model if given, and calls the evaluation function.
    """
    logger.info("Evaluating model...")

    # Only used for accessing the number of classes
    dataset = match_dataset("query")

    # num_classes not needed as classifier not used, but train classes is used to make compatibility with checkpoints easier
    model = init_model(cfg.MODEL.ARCH, num_classes=dataset.train_classes, two_branch=cfg.MODEL.TWO_BRANCH, device=device)
    model = model.to(device)

    if cfg.MODEL.CHECKPOINT:
        load_checkpoint(cfg.MODEL.CHECKPOINT, model)

    eval_model(model)


if __name__ == "__main__":
    main()

