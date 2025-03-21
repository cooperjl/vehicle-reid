import random

import numpy as np
import torch

from vehicle_reid import args
from vehicle_reid.utils import configure_logger



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    configure_logger()

    args.parse_command()

    set_seed(1234)
