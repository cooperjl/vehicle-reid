from .resnet import resnet50, resnet101
from .convnext import convnext_tiny, convnext_small
from .graph import GraphModule
from .siamese import Model

models = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'convnext-tiny': convnext_tiny,
    'convnext-small': convnext_small,
}

def init_model(name: str, num_classes: int, in_channels: int, two_branch: bool=True, pretrain=True):
    if name not in list(models.keys()):
        raise ValueError(f"model not found: {name}")

    if two_branch:
        return Model(models[name], num_classes, in_channels, pretrain)
    else:
        return models[name](num_classes, pretrain)

