import torch
import torch.nn as nn


class weighted_L1_loss(nn.Module):
    def __init__(self, weights, mode="sum"):
        super(weighted_L1_loss, self).__init__()
        self.weights = weights
        self.mode = mode

    def forward(self, gt, pred):
        out = torch.abs(gt - pred)
        out = self.weights * out
        if self.mode == "sum":
            loss = out.sum()
        elif self.mode == "mean":
            loss = out.mean()
        else:
            raise NotImplementedError
        return loss
