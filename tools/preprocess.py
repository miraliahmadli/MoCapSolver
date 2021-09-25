import torch

from tools.utils import xform_inv, xform_to_mat44, svd_rot

main_labels =\
        ['C7', 'CLAV', 'LANK', 'LBHD', 'LBWT', 'LELB', 'LFHD', 'LFIN', 
        'LFRM', 'LFWT', 'LHEE', 'LKNE', 'LMT5', 'LSHN', 'LSHO', 'LTHI', 
        'LTOE', 'LUPA', 'LWRA', 'LWRB', 'RANK', 'RBAC', 'RBHD', 'RBWT', 
        'RELB', 'RFHD', 'RFIN', 'RFRM', 'RFWT', 'RHEE', 'RKNE', 'RMT5', 
        'RSHN', 'RSHO', 'RTHI', 'RTOE', 'RUPA', 'RWRA', 'RWRB', 'STRN', 'T10']

main_joints = \
        ['root', 'lhipjoint', 'lfemur', 'ltibia', 'lfoot', 'ltoes',
        'rhipjoint', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 'lowerback',
        'upperback', 'thorax', 'lowerneck', 'upperneck', 'head', 'lclavicle',
        'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
        'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb']

local_frame_markers = [4, 9, 23, 28, 39, 40]
local_frame_joint = 11


def weight_assign(joint_to_marker_file, num_marker=41, num_joints=31):
    joint_to_marker = []
    with open(joint_to_marker_file, "r") as f:
        for l in f.readlines():
            splitted = l.split()[1:]
            joint_to_marker.append(splitted)

    w = torch.zeros((num_marker, num_joints))
    for i, markers in enumerate(joint_to_marker):
        for m in markers:
            w[main_labels.index(m), i] = 1

    return w


def get_Z(X, Y):
    '''
    Local offset computation function

    Parameters:
        X: global marker positions, dim: (n, m, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        Z: local offsets, dim: (n, m, j, 3)
    '''
    Y_inv = xform_inv(Y) # n x j x 3 x 4

    X_expand = torch.unsqueeze(X, axis = -1) # n x m x 3 x 1

    Y_inv_rot = torch.unsqueeze(Y_inv[..., : 3], 2) # n x m x 1 x 3 x 3
    Y_inv_tr = torch.unsqueeze(Y_inv[..., 3: ], 2) # n x m x 1 x 3 x 1

    Z = torch.matmul(Y_inv_rot, torch.unsqueeze(X_expand, 1)) + Y_inv_tr
    Z = Z.permute((0, 2, 1, 3, 4))
    Z = torch.squeeze(Z, -1)

    zeros = (X == 0.0)
    nans = zeros.any(axis=-1)

    Z[nans] = 0.0
    return Z


def clean_XY(X_read, Y_read, avg_bone_read, m_conv=0.56444, del_nans=False):#0.57803):
    '''
    Clean XYZ
​
    Parameters:
        X_read: global marker positions read from the npy, dim: (4, m, n)
        Y_read: rotation + translation matrices read from the npy, dim: (n, j, 3, 4)
        avg_bone_read: avg bone length
        m_conv: conversion constant
        del_nans: delete frames with nans
​
    Return:
        X: cleaned global marker positions, dim: (n, m, 3)
        Y: cleaned rotation + translation matrices, dim: (n, j, 3, 4)
        Z: local offsets, dim: (n, m, j, 3)
    '''
    X = torch.nan_to_num(X_read, nan=0.0, posinf=0.0, neginf=0.0)
    X = X.permute(2, 1, 0)
    X = X[..., : 3]

    zeros = (X == 0.0)
    nans = ~(zeros.any(dim=-1).any(dim=-1))

    avg_bone = avg_bone_read
    if del_nans:
        X = X[nans]
        avg_bone = avg_bone_read[nans]
    X *= (0.01 / (avg_bone[..., None] * m_conv))
    X = X[..., [1, 2, 0]]

    Y = Y_read.clone()
    if del_nans:
        Y = Y[nans]
    Y[..., 3] *= (1.0 / avg_bone[..., None])

    return X, Y, avg_bone


def local_frame(X, Y, X_mean):
    '''
    Local frame F calculation function
    rot: Rotation of local_frame_joint
    tr: Mean of translations of local_frame_markers

    Parameters:
        X: global marker positions, dim: (n, m, 3)
        X_mean: mean marker positions wrt reference joint, dim: (6, 1, 3)

    Return:
        F: Computed local frame, dim: (n, 3, 4)
        Y_local: Y fitted into local frame, dim: (n, j, 3, 4)
    '''
    X_picked = X[:, local_frame_markers, :]
    R, t = svd_rot(X_mean.permute(1, 2, 0), X_picked.permute(0, 2, 1))

    F = torch.cat([R, t], axis=-1)

    F_inv = xform_inv(F)
    F_inv_expand = torch.unsqueeze(F_inv, 1)

    Y_44 = xform_to_mat44(Y)
    Y_local = torch.matmul(F_inv_expand, Y_44)

    return F, Y_local

def local_frame_F(X, X_mean, local_frame_markers):
    X_picked = X[:, local_frame_markers, :]
    R, t = svd_rot(X_mean.permute(1, 2, 0), X_picked.permute(0, 2, 1))

    F = torch.cat([R, t], axis=-1)

    return F


if __name__ == "__main__":
    x = torch.rand(100, 41, 3)
    y = torch.rand(100, 41, 3)

    x_, y_ = local_frame(x, y)

    x = torch.Tensor(x).to("cuda")
    y = torch.Tensor(y).to("cuda")
    x_2, y_2 = local_frame(x, y)

    x_ = torch.Tensor(x_).to("cuda")
    y_ = torch.Tensor(y_).to("cuda")
    diff2 = torch.abs(x_ - x_2).sum().data
    diff3 = torch.abs(y_ - y_2).sum().data
    print("x", torch.max(torch.abs(x_ - x_2)))
    print("y", torch.max(torch.abs(y_ - y_2)))
    print("\nx", diff2, "\ny", diff3)
