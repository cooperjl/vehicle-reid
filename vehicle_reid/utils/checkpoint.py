import os

import torch

from vehicle_reid.config import cfg


def save_checkpoint(epoch: int, model_state_dict: dict, optimizer_state_dict: dict):
    """
    Saves the state of the model and optimizer for resumed training and model saving.

    Parameters
    ----------
    epoch : int
        the current epoch that has most recently finished.
    model_state_dict : dict
        the state dict of the model to be saved.
    optimizer_state_dict : dict
        the state dict of the optimizer to be saved.
    """
    if(not os.path.exists(cfg.MISC.SAVE_DIR)):
        os.mkdir(cfg.MISC.SAVE_DIR)

    file_name = "checkpoint_" + cfg.SOLVER.OPTIMIZER + f"_epoch-{epoch}" + ".pth"
    path = os.path.join(cfg.MISC.SAVE_DIR, file_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None):
    """
    Loads the state of the model and optimizer for resumed training and model testing.

    Parameters
    ----------
    path : str
        path to the file to be loaded.
    model : torch.nn.Module
        model to load the state of.
    optimizer : optim.Optimizer, optional
        Optimizer to load the state of, used only if resuming training.

    Returns
    -------
    start_epoch : int
        The epoch to resume training from, if used for resuming training.
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

def load_weights(model: torch.nn.Module, weights: dict):
    """
    Load weights into the model, allowing for layers of different sizes, for example the classifier.

    Parameters
    ----------
    model : torch.nn.Module
        model to load the state of.
    weights : dict
        state dict of weights to load.
    """
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

