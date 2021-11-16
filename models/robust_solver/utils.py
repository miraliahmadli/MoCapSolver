import torch
import torch.nn as nn
from models.loss import weighted_L1_loss

from models.vn_layers import *


class VNResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super(VNResidualBlock, self).__init__()
        self.bn = VNBatchNorm(hidden_size, 3)
        self.linear = VNLinear(hidden_size, hidden_size)
        self.lerelu = VNLeakyReLU(hidden_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.lerelu(out)
        x0 = self.linear(out)
        out = out + x0
        return out


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out += x
        out = self.relu(out)
        return out


class RS_loss(nn.Module):
    def __init__(self, weights_tr, weights_rot, mode="sum"):
        super(RS_loss, self).__init__()
        self.tr_loss = weighted_L1_loss(weights_tr, mode)
        self.rot_loss = weighted_L1_loss(weights_rot, mode)

    def forward(self, gt, pred):
        R, t = gt[..., :3], gt[..., 3:]
        R_hat, t_hat = pred[..., :3], pred[..., 3:]
        tr_loss = self.tr_loss(t, t_hat)
        rot_loss = self.rot_loss(R, R_hat)
        return rot_loss, tr_loss
