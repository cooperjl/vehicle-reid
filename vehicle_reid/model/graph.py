import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class GraphModule(nn.Module):
    def __init__(self, in_channels: int, device: str) -> None:
        super().__init__()

        # [num_edges, 2] tensor of directed edges. Final graph undirected as every edge repeated in both directions.
        # Stored this way for better readability; transposed into [2, num_edges] for PyTorch geometric.
        self.edge_index = (
            torch.tensor(
                [
                    [0, 1],  # f_global -> f_mid
                    [0, 2],  # f_global -> f_tl
                    [0, 3],  # f_global -> f_tr
                    [0, 4],  # f_global -> f_bl
                    [0, 5],  # f_global -> f_br
                    [1, 0],  # f_mid -> f_global
                    [1, 2],  # f_mid -> f_tl
                    [1, 3],  # f_mid -> f_tr
                    [1, 4],  # f_mid -> f_bl
                    [1, 5],  # f_mid -> f_br
                    [2, 0],  # f_tl -> f_global
                    [2, 1],  # f_tl -> f_mid
                    [2, 3],  # f_tl -> f_tr
                    [2, 4],  # f_tl -> f_bl
                    [3, 0],  # f_tr -> f_global
                    [3, 1],  # f_tr -> f_mid
                    [3, 2],  # f_tr -> f_tl
                    [3, 5],  # f_tr -> f_br
                    [4, 0],  # f_bl -> f_global
                    [4, 1],  # f_bl -> f_mid
                    [4, 2],  # f_bl -> f_tl
                    [4, 5],  # f_bl -> f_br
                    [5, 0],  # f_br -> f_global
                    [5, 1],  # f_br -> f_mid
                    [5, 3],  # f_br -> f_tr
                    [5, 4],  # f_br -> f_bl
                ],
                dtype=torch.long,
            )
            .t()
            .contiguous()
            .to(device)
        )

        # Configured separately for flexibility in changing, as previously was reduced, but this was replaced with dropout for
        # a more robust and accurate model.
        hidden_channels = in_channels

        self.graph_conv1 = GraphConv(
            in_channels=in_channels, out_channels=hidden_channels
        )
        self.graph_conv2 = GraphConv(
            in_channels=in_channels, out_channels=hidden_channels
        )

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
