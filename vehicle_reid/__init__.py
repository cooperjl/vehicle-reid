import random

import numpy as np
import torch

from vehicle_reid import args


def set_seed(seed):
    """Set the seeds to improve reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # False would be more reproducible, but has too negative an effect on performance.
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(1234)
    args.parse_command()

