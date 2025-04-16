import torch
from torchvision.models.resnet import (
    Bottleneck,
    ResNet,
    ResNet50_Weights,
    ResNet101_Weights,
)

from vehicle_reid.utils import load_weights


class FeatureResNet(ResNet):
    def __init__(
        self, classify: bool = True, pool: bool = True, device: str = "cpu", **kwargs
    ):
        super().__init__(
            block=Bottleneck, **{k: v for k, v in kwargs.items() if v is not None}
        )
        self.classify = classify
        self.pool = pool
        self.device = device

    def pool_and_flatten(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        local_f = x

        global_f = self.pool_and_flatten(x)

        f = global_f if self.pool else local_f

        if self.training and self.classify:
            c = self.fc(global_f)
            return c, f
        else:
            return f


def resnet50(
    num_classes: int,
    pretrain: bool = True,
    classify: bool = True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureResNet:
    model = FeatureResNet(
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    if pretrain:
        load_weights(model=model, weights=ResNet50_Weights.DEFAULT.get_state_dict())

    return model


def resnet101(
    num_classes: int,
    pretrain=True,
    classify=True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureResNet:
    model = FeatureResNet(
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    if pretrain:
        load_weights(model=model, weights=ResNet101_Weights.DEFAULT.get_state_dict())

    return model
