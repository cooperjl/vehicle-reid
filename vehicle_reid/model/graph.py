import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

EDGE_INDEX = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3],
    [1, 4], [1, 5], [2, 3], [2, 4], [3, 5], [4, 5], [1, 0], [2, 0], [3, 0], [4, 0], 
    [5, 0], [2, 1], [3, 1], [4, 1], [5, 1], [3, 2], [4, 2], [5, 3], [5, 4]], dtype=torch.long).t().contiguous().to("cuda:0")

class GraphModule(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        #self.fc1 = nn.Sequential(
        #    nn.Linear(in_channels, in_channels),
        #    nn.BatchNorm2d(in_channels),
        #)
        self.graph_conv = SAGEConv(in_channels, in_channels * 2)
        #self.fc2 = nn.Sequential(
        #    nn.Linear(in_channels * 2, in_channels),
        #    nn.BatchNorm2d(in_channels),
        #)

    def forward(self, x):
        #res = x
        #x = self.fc1(x)
        x = self.graph_conv(x, EDGE_INDEX)
        x = F.gelu(x)
        #x = F.fc2(x)
        #x = x + res

        output = torch.empty(0).to("cuda:0")
    
        for blah in x:
            output = torch.cat((output, blah), 1)
        return output

