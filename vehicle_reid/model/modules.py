import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck


class IBN(nn.Module):
    """
    Instance-Batch Normalisation layer from
    Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net
    <https://arxiv.org/pdf/1807.09441.pdf>

    Parameters
    ----------
    planes : int
        Number of channels for the input tensor.
    """

    def __init__(self, planes: int):
        super().__init__()
        half1 = int(planes / 2)
        half2 = planes - half1

        self.half = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BottleneckIBN(Bottleneck):
    """
    Simple extension for the Bottleneck class which overrides the first bn layer with IBN, for IBN-a.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ibn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(inplanes, planes, **kwargs)

        if ibn:
            self.bn1 = IBN(planes)
