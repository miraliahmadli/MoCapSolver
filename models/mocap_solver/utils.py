import torch
import torch.nn as nn

from models.loss import weighted_L1_loss
from tools.utils import LBS


class ResidualBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ResidualBlock, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.dense(out)
        out += x
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                add_offset=False, offset_dim=None):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.add_offset = add_offset and (offset_dim is not None)
        if self.add_offset:
            self.offset_enc = nn.Linear(offset_dim, hidden_size)

    def set_offset(self, offset):
        if not self.add_offset: raise Exception('Wrong Combination of Parameters')
        self.offset = offset.view(offset.shape[0], -1)

    def forward(self, x):
        out = self.dense(x)
        if self.add_offset:
            offset_res = self.offset_enc(self.offset)
            out += offset_res / 100
        return out


class MSBlock(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super(MSBlock, self).__init__()
        self.res_block = ResidualBlock(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.res_block(x)
        out = self.res_block(out)
        out = self.relu(out)
        out = self.dense(out)
        return out


'''
    X_c: marker configuration, local offset from each marker to each joint
        dim: (m, j, 3)
    X_t: template skeleton, offset of each joint rel. to its parent (in T-pose)
        dim: (j, 3)
    X_m: motion, actor's pose
        temporal sequence of local rotations of each joint relative to its parent’s coordinate frame in the kinematic chain, 
        and the global translation of the root joint 
        rotations are defined by quaternions
        dim: (Jx4 + 3)
'''
def FK(X_m, X_t):
    pass


def LBS(w, Y_c, Y_t):
    X = (Y_t.unsqueeze(0) + Y_c)*w # m x j x 3
    idx = torch.argmax(abs(X), -2, keepdim=True)
    X = X.gather(1, idx.view(-1, 1, 3))
    return X


class Motion_loss(nn.Module):
    def __init__(self, joint_weights, b1, b2):
        super(Motion_loss, self).__init__()
        self.b1 = b1
        self.b2 = b2
        self.l1_crit = weighted_L1_loss(joint_weights, mode="mean")
        self.fk_crit = weighted_L1_loss(joint_weights, mode="mean")

    def forward(self, Y_m, X_m, Y_t, X_t):
        loss_m = self.b1 * self.l1_crit(Y_m, X_m)
        loss_fk = self.b2 * self.fk_crit(FK(Y_m, Y_t), FK(X_m, X_t))
        loss = loss_m + loss_fk
        return loss


class Offset_loss(nn.Module):
    def __init__(self, joint_weights, b3, b4, weights):
        super(Motion_loss, self).__init__()
        self.b1 = b1
        self.b2 = b2
        self.w = weights
        self.l1_crit = weighted_L1_loss(joint_weights, mode="mean")
        self.lbs_crit = weighted_L1_loss(joint_weights, mode="mean")

    def forward(self, Y_c, X_c, Y_t, X_t):
        loss_c = self.b3 * self.l1_crit(Y_c, X_c)
        loss_lbs = self.b4 * self.lbs_crit(LBS(self.w, Y_c, Y_t), LBS(self.w, X_c, X_t))
        loss = loss_c + loss_lbs
        return loss


class AE_loss(nn.Module):
    def __init__(self, joint_weights, betas, weight_assignment):
        super(AE_loss, self).__init__()
        b1, b2, b3, b4 = betas
        self.crit_c = Offset_loss(joint_weights, b3, b4, weight_assignment)
        self.crit_t = weighted_L1_loss(joint_weights)
        self.crit_m = Motion_loss(joint_weights, b1, b2)

    def forward(self, Y, X):
        Y_c, Y_t, Y_m = Y
        X_c, X_t, X_m = X
        loss_c = self.crit_c(Y_c, X_c, Y_t, X_t)
        loss_t = self.crit_t(Y_t, X_t)
        loss_m = self.crit_m(Y_m, X_m, Y_t, X_t)
        return loss_t, loss_m, loss_c


class MS_loss(nn.Module):
    def __init__(self, joint_weights, alphas):
        super(MS_loss, self).__init__()
        self.a1, self.a2, self.a3, self.a4 = alphas
        self.crit = weighted_L1_loss(joint_weights)
        self.crit_c = weighted_L1_loss(joint_weights)
        self.crit_t = weighted_L1_loss(joint_weights)
        self.crit_m = weighted_L1_loss(joint_weights)

    def forward(self, Y, X):
        Y_, Y_c, Y_t, Y_m = Y
        X_, X_c, X_t, X_m = X
        loss_marker = self.crit(Y_, X_)
        loss_c = self.crit_c(Y_c, X_c)
        loss_t = self.crit_t(Y_t, X_t)
        loss_m = self.crit_m(Y_m, X_m)

        loss = self.a1 * loss_marker + self.a2 * loss_c + self.a3 * loss_t + self.a4 * loss_m
        return loss
