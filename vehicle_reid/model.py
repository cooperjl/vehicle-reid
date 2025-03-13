import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchvision.models.resnet import ResNet50_Weights, ResNet, Bottleneck


class GCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv()





class ModelBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None) -> None:
        super(ModelBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet2(nn.Module):
    def __init__(self, num_classes, block, n_layers_per_block: list, dropout=None):
        super(ResNet2, self).__init__()
        self.in_channels = 64
        self.feature_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, n_layers_per_block[0])
        self.layer2 = self.make_layer(block, 128, n_layers_per_block[1], stride=2)
        self.layer3 = self.make_layer(block, 256, n_layers_per_block[2], stride=2)
        self.layer4 = self.make_layer(block, 512, n_layers_per_block[3], stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def make_layer(self, block, channels, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, channels * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def forward(self, x):
        o = self.featuremaps(x)
        o = self.global_avgpool(o)
        o = o.view(o.size(0), -1)

        if not self.training:
            return o

        y = self.classifier(o) # extract classification for ce loss

        return y, o


class CustomResNet(ResNet):
    def __init__(self, layers: list[int], num_classes: int):
        super(CustomResNet, self).__init__(block=Bottleneck, layers=layers, num_classes=num_classes)

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
        x = torch.flatten(x, 1)

        if not self.training:
            return x

        c = self.fc(x)

        return c, x



def cresnet50(num_classes: int, pretrain=True):
    #model = ResNet2(
    #    num_classes=num_classes,
    #    block=ModelBlock,
    #    n_layers_per_block=
    #    dropout=None,
    #)

    model = CustomResNet(
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

