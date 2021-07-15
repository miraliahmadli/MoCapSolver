import torch
import numpy as np


'''
    Linear Blend Skinning function
    Parameters:
        w: weights associated with marker offsets, dim: (m, j)
        Z: local offsets, dim: (n, m, j, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)
    Return:
        X: global marker positions, dim: (n, m, 3)
'''
def LBS_np(w, Z, Y):
    m, j = w.shape
    n = Z.shape[0]

    w_ = w.transpose(1, 0) # j x m

    z_padding = np.ones((n, m, j, 1))
    Z_ = np.concatenate((Z, z_padding), axis=3)
    Z_ = Z_.transpose(2, 0, 3, 1) # j x n x 4 x m

    Y_ = Y.transpose(1, 0, 2, 3) # j x n x 3 x 4

    prod = np.matmul(Y_, Z_).transpose(0, 3, 1, 2) # j x m x n x 3

    # for i in range(j):
    #     for k in range(m):
    #         prod[i, k, :, :] *= w_[i, k]
    # X = np.sum(prod, axis=0).permute(1, 0, 2) # n x m x 3

    X = np.sum(
            np.multiply(prod, w_.reshape((j, m, 1, 1))), 
            axis=0).transpose(1, 0, 2) # n x m x 3

    # for loop version
    # X_1 = np.zeros((n, m, 3))
    # for i in range(j):
    #     w_i = w[:, i] # m x 1
    #     Z_i = Z[:, :, i, :] # n x m x 3
    #     Y_i = Y[:, i, :, :] # n x 3 x 4

    #     prod = np.zeros((n, m, 3))
    #     for pose in range(n):
    #         for k in range(3):
    #             for t in range(m):
    #                 prod[pose, t, k] = Y_i[pose, k, 3] +\
    #                                     Y_i[pose, k, 0]*Z_i[pose, t, 0] +\
    #                                     Y_i[pose, k, 1]*Z_i[pose, t, 1] +\
    #                                     Y_i[pose, k, 2]*Z_i[pose, t, 2]
    #     for t in range(m):
    #         prod[:, t, :] *= w_i[t]
        
    #     X_1 += prod

    # max_error = float("-inf")
    # for pose in range(n):
    #     for k in range(3):
    #         for t in range(m):
    #             max_error = max(max_error, abs(X[pose, t, k] - X_1[pose, t, k]))
    # print(max_error)
    # print((X == X_1).all())
    return X


def LBS_torch(w, Z, Y):
    m, j = w.shape
    n = Z.shape[0]

    w_ = w.permute(1, 0) # j x m

    z_padding = torch.ones((n, m, j, 1))
    Z_ = torch.cat((Z, z_padding), axis=3)
    Z_ = Z_.permute(2, 0, 3, 1) # j x n x 4 x m

    Y_ = Y.permute(1, 0, 2, 3) # j x n x 3 x 4

    prod = torch.matmul(Y_, Z_).permute(0, 3, 1, 2) # j x m x n x 3

    X = torch.sum(
            torch.mul(prod, w_.reshape((j, m, 1, 1))), 
            axis=0).permute(1, 0, 2) # n x m x 3

    return X


'''
    Implementation of "Least-Squares Rigid Motion Using SVD"
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    Problem: 
        finding rotation R and translation t matrices
        that minimizes \sum (w_i ||(Rp_i + t) - q_i||^2) 
        (least square error)

    Solution:
        t = q_mean - R*p_mean
        R = V * D * U.T
'''
def svd_rot(P, Q, w):
    assert P.shape == Q.shape
    n, k = P.shape
    assert k == w.shape[0]
    assert w.shape[0] == w.shape[1]

    # X,Y are n x k
    P_ = P.mean(axis=0)
    Q_ = Q.mean(axis=0)
    X = P - P_
    Y = Q - Q_

    # S is n x n
    S = np.matmul(np.matmul(X, W), Y.T)

    # U, V are n x m
    U, _, V_t = np.linalg.svd(S)
    V = V_t.T

    m = U.shape[1]
    D = np.diag(np.ones(m))
    D[-1, -1] = np.linalg.det(V*U.T)

    # R is n x n
    R = np.matmul(np.matmul(V, D), U.T)

    # t is n x k
    t = Q_ - np.matmul(R, P_)

    return R, t


def test_lbs():
    n = 20
    j = 31
    m = 41

    w = np.random.rand(m, j)
    Z = np.random.rand(n, m, j, 3)
    Y = np.random.rand(n, j, 3, 4)

    # w = np.ones((m, j))
    # Z = np.ones((n, m, j, 3))*15.4
    # Y = np.ones((n, j, 3, 4))*30.3

    # Z = np.arange(n*m*j*3).reshape((n, m, j, 3))
    # Y = np.arange(n*j*12).reshape((n, j, 3, 4))


    w = torch.Tensor(w)
    Z = torch.Tensor(Z)
    Y = torch.Tensor(Y)
    # X = LBS_np(w, Z, Y)
    X = LBS_torch(w, Z, Y)
    print(X.shape)


def test_svd():
    pass


if __name__ == "__main__":
    test_lbs()
    # test_svd()
