import torch
import numpy as np

from tools.utils import xform_inv_np, xform_inv_torch, xform_to_mat44_np, xform_to_mat44_torch

main_labels =\
        ['C7', 'CLAV', 'LANK', 'LBHD', 'LBWT', 'LELB', 'LFHD', 'LFIN', 
        'LFRM', 'LFWT', 'LHEE', 'LKNE', 'LMT5', 'LSHN', 'LSHO', 'LTHI', 
        'LTOE', 'LUPA', 'LWRA', 'LWRB', 'RANK', 'RBAC', 'RBHD', 'RBWT', 
        'RELB', 'RFHD', 'RFIN', 'RFRM', 'RFWT', 'RHEE', 'RKNE', 'RMT5', 
        'RSHN', 'RSHO', 'RTHI', 'RTOE', 'RUPA', 'RWRA', 'RWRB', 'STRN', 'T10']

local_frame_markers = [4, 9, 23, 28, 39, 40]
local_frame_joint = 11


def weight_assign(joint_to_marker_file, num_marker=41, num_joints=31):
    joint_to_marker = []
    with open(joint_to_marker_file, "r") as f:
        for l in f.readlines():
            splitted = l.split()[1:]
            joint_to_marker.append(splitted)

    w = np.zeros((num_marker, num_joints))
    for i, markers in enumerate(joint_to_marker):
        for m in markers:
            w[main_labels.index(m), i] = 1

    return w


def get_Z_np(X, Y):
    '''
    Local offset computation function

    Parameters:
        X: global marker positions, dim: (n, m, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        Z: local offsets, dim: (n, m, j, 3)
    '''
    Y_inv = xform_inv_np(Y) # n x j x 3 x 4

    X_expand = np.expand_dims(X, axis = -1) # n x m x 3 x 1

    Y_inv_rot = np.expand_dims(Y_inv[..., : 3], 2) # n x m x 1 x 3 x 3
    Y_inv_tr = np.expand_dims(Y_inv[..., 3: ], 2) # n x m x 1 x 3 x 1

    Z = np.matmul(Y_inv_rot, np.expand_dims(X_expand, 1)) + Y_inv_tr
    Z = Z.transpose((0, 2, 1, 3, 4))
    Z = np.squeeze(Z)
    return Z


def get_Z_torch(X, Y):
    Y_inv = xform_inv_torch(Y) # n x j x 3 x 4

    X_expand = torch.unsqueeze(X, axis = -1) # n x m x 3 x 1

    Y_inv_rot = torch.unsqueeze(Y_inv[..., : 3], 2) # n x m x 1 x 3 x 3
    Y_inv_tr = torch.unsqueeze(Y_inv[..., 3: ], 2) # n x m x 1 x 3 x 1

    Z = torch.matmul(Y_inv_rot, torch.unsqueeze(X_expand, 1)) + Y_inv_tr
    Z = Z.permute((0, 2, 1, 3, 4))
    Z = torch.squeeze(Z)
    return Z


def clean_XYZ_np(X_read, Y_read, avg_bone, m_conv=0.056444):
    '''
    Clean XYZ

    Parameters:
        X_read: global marker positions read from the npy, dim: (4, m, n)
        Y: rotation + translation matrices read from the npy, dim: (n, j, 3, 4)

    Return:
        X: cleaned global marker positions, dim: (n, m, 3)
        Y: cleaned rotation + translation matrices, dim: (n, j, 3, 4)
        Z: local offsets, dim: (n, m, j, 3)
    '''
    nans = np.isnan(X_read)[0].transpose(1, 0)
    nans_float = 1 - nans.astype(float)
    nans_expand = np.expand_dims(nans_float, 2)
    nans_expand2 = np.expand_dims(nans_expand, 3)
    
    avg_bone_m = avg_bone * m_conv

    X = np.nan_to_num(X_read)
    X = X.transpose(2, 1, 0)
    X *= nans_expand
    X = X[..., : 3]
    X *= (1.0 / (avg_bone_m)) * 0.001
    X = X[..., [1, 2, 0]]

    Y = np.copy(Y_read)
    Y[..., 3] *= (1.0 / avg_bone)
    
    Z = get_Z_np(X, Y)
    Z *= nans_expand2

    return X, Y, Z


def clean_XYZ_torch(X_read, Y_read, avg_bone, m_conv=0.056444):
    nans = torch.isnan(X_read)[0].transpose(1, 0)
    nans_float = 1 - nans.to(torch.float32)
    nans_expand = torch.unsqueeze(nans_float, 2)
    nans_expand2 = torch.unsqueeze(nans_expand, 3)
    
    avg_bone_m = avg_bone * m_conv

    X = torch.nan_to_num(X_read)
    X = X.permute(2, 1, 0)
    X *= nans_expand
    X = X[..., : 3]
    X *= (1.0 / (avg_bone_m)) * 0.001
    X = X[..., [1, 2, 0]]

    Y = Y_read.clone()
    Y[..., 3] *= (1.0 / avg_bone)
    
    Z = get_Z_torch(X, Y)
    Z *= nans_expand2

    return X, Y, Z


def local_frame_np(X, Y):
    '''
    Local frame F calculation function
    rot: Rotation of local_frame_joint
    tr: Mean of translations of local_frame_markers

    Parameters:
        X: global marker positions, dim: (n, m, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        F: Computed local frame, dim(n, 3, 4)
        Y_local: Y fitted into local frame, dim: (n, j, 3, 4)
    '''
    X_tr = np.mean(X[:, local_frame_markers, :], axis = 1)
    X_rot = Y[:, local_frame_joint, :, :3]

    F = np.concatenate([X_rot, np.expand_dims(X_tr, 2)], axis = 2)

    F_inv = xform_inv_np(F)
    F_inv_expand = np.expand_dims(F_inv, 1)

    Y_44 = xform_to_mat44_np(Y)
    Y_local = np.matmul(F_inv_expand, Y_44)

    return F, Y_local


def local_frame_torch(X, Y, device="cuda"):
    X_tr = torch.mean(X[:, local_frame_markers, :], axis = 1)
    X_rot = Y[:, local_frame_joint, :, :3]

    F = torch.cat((X_rot, torch.unsqueeze(X_tr, 2)), axis = 2)

    F_inv = xform_inv_torch(F)
    F_inv_expand = torch.unsqueeze(F_inv, 1)

    Y_44 = xform_to_mat44_torch(Y, device)
    Y_local = torch.matmul(F_inv_expand, Y_44)

    return F, Y_local


if __name__ == "__main__":
    x = np.random.rand(100, 41, 3)
    y = np.random.rand(100, 31, 3, 4)

    x_, y_ = local_frame_np(x, y)

    x = torch.Tensor(x).to("cuda")
    y = torch.Tensor(y).to("cuda")
    x_2, y_2 = local_frame_torch(x, y)

    x_ = torch.Tensor(x_).to("cuda")
    y_ = torch.Tensor(y_).to("cuda")
    diff2 = torch.abs(x_ - x_2).sum().data
    diff3 = torch.abs(y_ - y_2).sum().data
    print("x", torch.max(torch.abs(x_ - x_2)))
    print("y", torch.max(torch.abs(y_ - y_2)))
    print("\nx", diff2, "\ny", diff3)
