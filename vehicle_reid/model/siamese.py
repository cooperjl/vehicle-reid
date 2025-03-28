import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

from vehicle_reid.config import cfg
from vehicle_reid.model import GraphModule


class Model(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int, in_channels: int, pretrain: bool=True) -> None:
        super().__init__()
            
        #self.global_model = init_model(num_classes, pretrain=True)
        #self.local_model = init_model(num_classes, pretrain=True)
       # global_model = global_model.to(device)

        self.global_model = model(num_classes, pretrain)
        self.local_model = model(num_classes, pretrain)

        self.five_crop = transforms.FiveCrop(size=(int(cfg.INPUT.WIDTH / 3), int(cfg.INPUT.HEIGHT / 3)))

        #self.model = model
        self.local_model.classify = False

        self.graph = GraphModule(in_channels=in_channels)


    def forward_once(self, x) -> torch.Tensor: # rename this TODO
        output = self.local_model(x)
        return output

    def forward(self, x: torch.Tensor):
        # Local inputs
        (x_tl, x_tr, x_bl, x_br, x_mid) = self.five_crop(x)

        # Global model
        if self.training:
            c, f_global = self.global_model(x)
        else:
            f_global = self.global_model(x)

        # Siamese feature extraction
        f_mid = self.forward_once(x_mid)
        f_tl = self.forward_once(x_tl)
        f_tr = self.forward_once(x_tr)
        f_bl = self.forward_once(x_bl)
        f_br = self.forward_once(x_br)

    
        # 2 layer GCN layers
        feature_stack = torch.stack((f_global, f_mid, f_tl, f_tr, f_bl, f_br), dim=0)

        #output = torch.cat((f_mid, f_tl, f_tr, f_bl, f_br), 1)
        

        output = self.graph(feature_stack)

        if self.training:
            return c, output
        else:
            return output

