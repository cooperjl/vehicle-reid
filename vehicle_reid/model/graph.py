import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class GraphModule(nn.Module):
    def __init__(self, in_channels: int, device: str) -> None:
        super().__init__()

        # [2, num_edges] array, effectively reversed in second index to make undirected.
        self.edge_index = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3],
            [1, 4], [1, 5], [2, 3], [2, 4], [3, 5], [4, 5], [1, 0], [2, 0], [3, 0], [4, 0], 
            [5, 0], [2, 1], [3, 1], [4, 1], [5, 1], [3, 2], [4, 2], [5, 3], [5, 4]], dtype=torch.long).t().contiguous().to(device)

        # Configured separately for flexibility in changing, as previously was reduced, but this was replaced with dropout for
        # a more robust and accurate model.
        hidden_channels = in_channels

        self.graph_conv1 = GraphConv(in_channels=in_channels, out_channels=hidden_channels)
        self.graph_conv2 = GraphConv(in_channels=in_channels, out_channels=hidden_channels)

        self.dropout = nn.Dropout(p=0.5)

        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.batch_norm.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.graph_conv1(self.dropout(x), self.edge_index)
        x = F.gelu(x)
        x = self.graph_conv2(self.dropout(x), self.edge_index)
        x = F.gelu(x)

        # Batch mean pooling and batch norm
        output = x.permute(1, 0, 2).contiguous()
        output = torch.mean(output, 1)
        output = self.dropout(output)

        output = output.view(-1, output.size()[1])

        output = self.batch_norm(output)

        return output

