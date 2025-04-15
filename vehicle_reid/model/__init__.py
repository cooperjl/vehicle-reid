from .resnet import resnet50, resnet101
from .convnext import convnext_tiny, convnext_small
from .model import Model

models = {
    'resnet50': {'model': resnet50, 'features': 2048},
    'resnet101': {'model': resnet101, 'features': 2048},
    'convnext-tiny': {'model': convnext_tiny, 'features': 768},
    'convnext-small': {'model': convnext_small, 'features': 768},
}

def init_model(name: str, num_classes: int=None, two_branch: bool=True, pretrain: bool=True, device: str="cpu"):
    """
    Initialises the model, depending on configuration options.

    Parameters
    ----------
    name : str
        The name of the architecture for the feature extraction model. Valid options visable in models dict keys in this file.
    num_classes : int, optional
        The number of training classes for the classifier in training with cross entropy loss. The default value is None, for
        use outside of training.
    two_branch : bool, optional
        Whether to use the two-branch graph-based model. The default value is True.
    pretrain : bool, optional
        Whether to use pretrained weights for the feature extraction model. The default value is True.
    device : str, optional
        What device the model will be on. Used for storage for access from other locations, need to move the model manually after
        initialising. The default value is "cpu", but most likely this will be changed.

    Returns
    -------
    model : nn.Module
        The model, either siamese.Model type, or a feature extraction model, depending on two_branch parameter.
    """
    if name not in list(models.keys()):
        raise ValueError(f"model not found: {name}")
    
    model_dict = models[name]

    if two_branch:
        return Model(model_dict['model'], in_channels=model_dict['features'], num_classes=num_classes, pretrain=pretrain, device=device)
    else:
        return model_dict['model'](num_classes=num_classes, pretrain=pretrain, device=device)

