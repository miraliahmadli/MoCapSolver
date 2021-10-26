import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.HUBER_DELTA = delta

    def forward(self, input, target):
        error_mat = input - target
        _error_ = torch.sqrt(torch.sum(error_mat **2))
        HUBER_DELTA = self.HUBER_DELTA
        switch_l = _error_<HUBER_DELTA
        switch_2 = _error_>=HUBER_DELTA
        x = switch_l * (0.5* _error_**2 ) + switch_2 * (0.5* HUBER_DELTA**2 + HUBER_DELTA*(_error_-HUBER_DELTA))
        return x


class weighted_L1_loss(nn.Module):
    def __init__(self, weights, mode="sum", loss_fn="SmoothL1"):
        super(weighted_L1_loss, self).__init__()
        self.weights = weights
        if loss_fn == "Huber":
            self.crit = HuberLoss()
        elif loss_fn == "SmoothL1":
            self.crit = nn.SmoothL1Loss(reduction=mode)
        elif loss_fn == "MSE":
            self.crit = nn.MSELoss()
        else:
            self.crit = nn.L1Loss(reduction=mode)

    def forward(self, gt, pred):
        input = self.weights * pred
        target = self.weights * gt
        loss = self.crit(input, target)
        return loss
