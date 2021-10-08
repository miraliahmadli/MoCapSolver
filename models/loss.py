import torch
import torch.nn as nn


class weighted_L1_loss(nn.Module):
    def __init__(self, weights, mode="sum"):
        super(weighted_L1_loss, self).__init__()
        self.weights = weights
        self.crit = nn.HuberLoss(reduction=mode)
        # self.crit = nn.SmoothL1Loss(reduction=mode)
        # self.crit = nn.L1Loss(reduction=mode)

    def forward(self, gt, pred):
        input = self.weights * pred
        target = self.weights * gt
        loss = self.crit(input, target)
        return loss
