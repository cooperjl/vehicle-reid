import math

import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

from .graph import GraphModule


class Model(nn.Module):
    def __init__(self, model: nn.Module, in_channels: int, num_classes: int, device: str="cpu", pretrain: bool=True) -> None:
        super().__init__()
        
        self.device = device
        self.feature_model = model(num_classes, pretrain, pool=False, device=device)
        self.graph = GraphModule(in_channels=in_channels, device=device)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            _, output = self.feature_model(x)
        else:
            output = self.feature_model(x)
        return output

    def crop_and_pool(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        crop_size = [math.ceil(x.shape[-1] * (2/3)), math.ceil(x.shape[-2] * (2/3))]

        features = (x,) + transforms.functional.five_crop(x, size=crop_size)
        features = tuple(self.feature_model.pool_and_flatten(f) for f in features)

        return features

    def forward(self, x: torch.Tensor):
        # Global features
        c = None
        if self.training:
            c, features = self.feature_model(x)
        else:
            features = self.feature_model(x)

        # Extract local features, and pool + flatten global and local features
        features = self.crop_and_pool(features)

        # Feature stack for graph
        feature_stack = torch.stack(features, dim=0)

        # Graph network
        output = self.graph(feature_stack)

        if c is not None:
            return c, output
        else:
            return output

