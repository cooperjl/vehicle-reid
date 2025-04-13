import torch
import torch.nn as nn

# For consistent importing
CrossEntropyLoss = nn.CrossEntropyLoss

class TripletLoss(nn.Module):
    def __init__(self, margin: float=1.0):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        inputs : torch.Tensor
            feature matrix with shape (batch_size, feat_dim).
        targets : torch.Tensor
            target labels with shape (num_classes).

        Returns
        -------
        loss : torch.Tensor
            triplet loss for use in training.
        """
        n = inputs.size(0)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = torch.addmm(dist, inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

