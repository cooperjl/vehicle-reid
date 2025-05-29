import logging

import torch

from vehicle_reid.config import cfg
from vehicle_reid.eval import eval_model
from vehicle_reid.model import init_model
from vehicle_reid.utils import load_checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def test_model():
    """
    Main testing function, which initialises the model, loads a model if given, and calls the evaluation function.
    """
    logger.info("Evaluating model...")

    model = init_model(cfg.MODEL.ARCH, two_branch=cfg.MODEL.TWO_BRANCH, device=device)
    model = model.to(device)

    if cfg.MODEL.CHECKPOINT:
        load_checkpoint(cfg.MODEL.CHECKPOINT, model)

    model.eval()

    rerank = cfg.DATASET.NAME == "veri"  # Only apply re-ranking on VeRi-776
    eval_model(model, rerank=rerank)
