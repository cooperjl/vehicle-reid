import torch.optim as optim

def init_optimizer(
        optim_str: str,
        params,
        lr=0.005, # learning rate
        weight_decay=5e-4, # weight decay
        momentum=0.9, # momentum factor for sgd and rmsprop
        ) -> optim.Optimizer:
    match optim_str:
        case "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        case "sgd":
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        case _:
            raise ValueError(f"invalid optimizer: {optim_str}")
