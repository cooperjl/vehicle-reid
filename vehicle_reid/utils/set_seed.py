import random

import numpy as np
import torch

from vehicle_reid.config import cfg

def set_seed():
    """Set the seeds to improve reproducibility."""
    if cfg.MISC.SEED is None:
        return

    torch.manual_seed(cfg.MISC.SEED)
    random.seed(cfg.MISC.SEED)
    np.random.seed(cfg.MISC.SEED)
    # False would be more reproducible, but has too negative an effect on performance.
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True

