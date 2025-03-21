from .resnet import resnet50
from .convnext import convnext_tiny

models = {
    'resnet50': resnet50,
    'convnext-tiny': convnext_tiny,
}

def init_model(name: str, num_classes: int, pretrain=True):
    if name not in list(models.keys()):
        raise ValueError(f"model not found: {name}")

    return models[name](num_classes, pretrain=pretrain)

