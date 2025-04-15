import torch
import torch.optim as optim

from vehicle_reid.config import cfg


def init_optimizer(named_parameters) -> optim.Optimizer:
    """
    Function which uses the configuration file to configure an optimiser for the given parameters.

    Parameters
    ----------
    named_parameters : Iterator[Tuple[str, torch.nn.Parameter]]
        the named parameters of the model.

    Returns
    -------
    optimizer : optim.Optimizer
        the configured optimizer, using the model parameters and configuration file.
    """
    if cfg.SOLVER.DECAY_BN_BIAS:
        params = named_parameters
    else:
        params = get_param_groups(named_parameters)
    
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

def get_param_groups(named_parameters) -> list[dict[str, float | torch.nn.Parameter]]:
    """
    Function which sets weight decay, to 0.0 on batch norm and bias layers, using parameter groups.

    Parameters
    ----------
    named_parameters : Iterator[tuple[str, torch.nn.Parameter]]
        the named parameters of the model.

    Returns
    -------
    param_groups : list[dict[str, float | torch.nn.Parameter]]
        list of dicts for the configured parameters for the optimiser, one dict for no decay
        and the other for decay. the keys for each are "weight_decay" and "params".
    """
    param_group_dict = {}

    for name, param in named_parameters:
        if not param.requires_grad:
            continue # continue on frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            # disable decay on batch norm and bias layers
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if group_name not in param_group_dict:
            param_group_dict[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }

        param_group_dict[group_name]["params"].append(param)
    
    return list(param_group_dict.values())

