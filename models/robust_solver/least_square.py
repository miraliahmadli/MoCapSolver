import numpy as np
import torch
import torch.nn as nn

from tools.preprocess import weight_assign
from tools.utils import svd_rot as svd_solver


class LS_solver(nn.Module):
    """
    Least Square solver
    """
    def __init__(self, num_joints: int, weights):
        super(LS_solver, self).__init__()
        self.num_joints = num_joints
        self.w = weights
        self.solver = svd_solver

    def forward(self, X, Z):
        output_size = (X.shape[0], self.num_joints, 3, 4)
        Y_hat = torch.empty(output_size).to(torch.float32).to(X.device) # n x j x 3 x 4
        for i in range(self.num_joints):
            markers = (self.w[:, i] == 1).nonzero(as_tuple=False).view((-1))
            Z_ = Z[:, markers, i].permute(0, 2, 1) # n x 3 x m
            X_ = X[:, markers].permute(0, 2, 1) # n x 3 x m
            R, t = self.solver(Z_, X_) # n x 3 x 3, n x 3 x 1
            R_t = torch.cat((R, t), -1)
            Y_hat[:, i] = R_t
        return Y_hat


def test():
    num_markers = 41
    num_joints = 31
    batch_size = 100

    w = weight_assign("dataset/joint_to_marker_one2one.txt", num_markers, num_joints)
    model = LS_solver(num_joints, w, "cuda")
    print(model)
    x = torch.randn(batch_size, num_markers, 3)
    z = torch.randn(batch_size, num_markers, num_joints, 3)


    y_hat = model(x, z)
    print(y_hat.shape)

if __name__ == "__main__":
    test()
