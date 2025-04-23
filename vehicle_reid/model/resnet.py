import torch
from torchvision.models.resnet import (
    Bottleneck,
    ResNet,
    ResNet50_Weights,
    ResNet101_Weights,
)

import torch.nn as nn
from vehicle_reid.utils import load_weights

from .modules import BottleneckIBN


class FeatureResNet(ResNet):
    def __init__(
        self,
        block: Bottleneck = Bottleneck,
        classify: bool = True,
        pool: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            block=block, **{k: v for k, v in kwargs.items() if v is not None}
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


class FeatureResNetIBN(FeatureResNet):
    def __init__(self, **kwargs):
        super().__init__(block=BottleneckIBN, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        **_,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        ibn = planes != 512
        layers = []
        layers.append(
            block(self.inplanes, planes, ibn, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)


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


def resnet50_ibn_a(
    num_classes: int,
    pretrain: bool = True,
    classify: bool = True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureResNet:
    model = FeatureResNetIBN(
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    weights = torch.hub.load_state_dict_from_url(
        "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
    )

    if pretrain:
        load_weights(model=model, weights=weights)

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


def resnet101_ibn_a(
    num_classes: int,
    pretrain: bool = True,
    classify: bool = True,
    pool: bool = True,
    device: str = "cpu",
) -> FeatureResNet:
    model = FeatureResNetIBN(
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        classify=classify,
        pool=pool,
        device=device,
    )

    weights = torch.hub.load_state_dict_from_url(
        "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth"
    )

    if pretrain:
        load_weights(model=model, weights=weights)

    return model
