import torch
from torchvision.models.convnext import ConvNeXt, CNBlockConfig, ConvNeXt_Tiny_Weights

class FeatureConvNeXt(ConvNeXt):
    def __init__(
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float,
        num_classes: int,
        classify: bool=True,
    ) -> None:
        super().__init__(block_setting, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes)

        self.classify = classify

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)

        f = torch.flatten(x, 1)

        if not self.training or not self.classify:
            return f

        # flatten included in classifier, so use x not f
        c = self.classifier(x)

        return c, f


def convnext_tiny(num_classes: int, pretrain: bool=True, classify: bool=True) -> FeatureConvNeXt:
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
        classify=classify
    )

    if pretrain:
        pretrain_weights = ConvNeXt_Tiny_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_weights.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    return model

def convnext_small(num_classes: int, pretrain: bool=True, classify: bool=True) -> FeatureConvNeXt:
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
        classify=classify
    )

    if pretrain:
        pretrain_weights = ConvNeXt_Tiny_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_weights.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    return model

