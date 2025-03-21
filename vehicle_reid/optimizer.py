import torch.optim as optim

from vehicle_reid.config import cfg

def init_optimizer(
    params,
) -> optim.Optimizer:

    match cfg.SOLVER.OPTIMIZER:
        case "adamw":
            return optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        case "adam":
            return optim.Adam(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        case "sgd":
            return optim.SGD(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                             momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
        case _:
            raise ValueError(f"invalid optimizer: {cfg.SOLVER.OPTIMIZER}")
