import torch
from torchvision.models.resnet import ResNet, Bottleneck, ResNet50_Weights


class FeatureResNet(ResNet):
    def __init__(self, layers: list[int], num_classes: int):
        super().__init__(block=Bottleneck, layers=layers, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        f = torch.flatten(x, 1)

        if not self.training:
            return f

        c = self.fc(f)

        return c, f


def resnet50(num_classes: int, pretrain=True):
    model = FeatureResNet(
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
    )

    if pretrain:
        pretrain_weights = ResNet50_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_weights.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    return model

