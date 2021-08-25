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


class Holden_loss(nn.Module):
    def __init__(self, weights_tr, weights_rot, mode="sum"):
        super(Holden_loss, self).__init__()
        self.tr_loss = weighted_L1_loss(weights_tr, mode)
        self.rot_loss = weighted_L1_loss(weights_rot, mode)

    def forward(self, gt, pred):
        R, t = gt[..., :3], gt[..., 3:]
        R_hat, t_hat = pred[..., :3], pred[..., 3:]
        tr_loss = self.tr_loss(t, t_hat)
        rot_loss = self.rot_loss(R, R_hat)
        return rot_loss, tr_loss
