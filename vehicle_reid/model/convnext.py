import torch
from torchvision.models.convnext import (
    CNBlockConfig,
    ConvNeXt,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
)

from vehicle_reid.utils import load_weights


class FeatureConvNeXt(ConvNeXt):
    def __init__(
        self, classify: bool = True, pool: bool = True, device: str = "cpu", **kwargs
    ):
        super().__init__(**{k: v for k, v in kwargs.items() if v is not None})

        self.classify = classify
        self.pool = pool
        self.device = device

    def pool_and_flatten(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor):
        x = self.features(x)

        local_f = x
        global_f = self.avgpool(x)  # flatten included in classifier, so don't flatten

        f = global_f if self.pool else local_f

        if self.training and self.classify:
            c = self.classifier(global_f)
            return c, f
        else:
            return f


def convnext_tiny(
    num_classes: int,
    pretrain: bool = True,
    classify: bool = True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]

    model = FeatureConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.1,
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    if pretrain:
        load_weights(
            model=model, weights=ConvNeXt_Tiny_Weights.DEFAULT.get_state_dict()
        )

    return model


def convnext_small(
    num_classes: int,
    pretrain: bool = True,
    classify: bool = True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]

    model = FeatureConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.1,
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    if pretrain:
        load_weights(
            model=model, weights=ConvNeXt_Small_Weights.DEFAULT.get_state_dict()
        )

    return model
