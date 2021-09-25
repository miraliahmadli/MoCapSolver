import torch
import numpy as np


def xform_to_mat44(X):
    '''
    Converts 3 x 4 rigid body tranformations to 4 x 4

    Parameters:
        X: Rigid body transformation matrix, dim: (..., 3, 4)

    Return:
        X_44: 4 x 4 rigid body transformation matrix, dim: (..., 4, 4)
    '''
    shape = X.shape[: -2] + (1, 4)
    affine = torch.zeros(shape, device=X.device)
    affine[..., -1] = 1
    X_44 = torch.cat((X, affine), axis = -2)

    return X_44


def xform_inv(Y):
    '''
    Inverse of the rigid body transformation

    Parameters:
        Y: Rigid body transformation matrix, dim: (..., 3, 4)

    Return:
        Y_inv: Inverse of rigid body transformation matrix, dim: (..., 3, 4)
    '''
    T, R = Y[..., 3: ], Y[..., :3]
    R_inv = R.transpose(-1, -2)
    T_inv = -T

    R_x_T = torch.matmul(R_inv, T_inv)
    Y_inv = torch.cat((R_inv, R_x_T), axis = -1)

    return Y_inv


def LBS(w, Y, Z):
    '''
    Linear Blend Skinning function

    Args:
        w: weights associated with marker offsets, dim: (m, j)
        Z: local offsets, dim: (n, m, j, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        X: global marker positions, dim: (n, m, 3)
    '''
    m, j = w.shape
    n = Z.shape[0]

    w_ = w.permute(1, 0) # j x m

    z_padding = torch.ones((n, m, j, 1), device=Z.device)
    Z_ = torch.cat((Z, z_padding), axis=3)
    Z_ = Z_.permute(2, 0, 3, 1) # j x n x 4 x m

    Y_ = Y.permute(1, 0, 2, 3) # j x n x 3 x 4

    prod = torch.matmul(Y_, Z_).permute(0, 3, 1, 2) # j x m x n x 3

    X = torch.sum(
            torch.mul(prod, w_.reshape((j, m, 1, 1))), 
            axis=0).permute(1, 0, 2) # n x m x 3

    return X


def svd_rot(P, Q):
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
    assert P.shape[-2:] == Q.shape[-2:]
    d, n = P.shape[-2:]

    # X,Y are d x n
    P_ = torch.sum(P, axis=-1) / n
    Q_ = torch.sum(Q, axis=-1) / n
    X = P - P_[..., None]
    Y = Q - Q_[..., None]
    Yt = Y.permute(0, 2, 1)

    # S is d x d
    S = torch.matmul(X, Yt)

    # U, V are d x d
    U, _, V = torch.svd(S)
    # V = V_t.permute(0, 2, 1)
    Ut = U.permute(0, 2, 1)

    det = torch.det(torch.matmul(V, Ut))
    Ut[:, -1, :] *= det.view(-1, 1)

    # R is d x d
    R = torch.matmul(V, Ut)

    # t is d x n
    t = Q_.view(-1, d, 1) - torch.matmul(R, P_.view(-1, d, 1))

    return R, t


def symmetric_orthogonalization(x):
  """
  Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r


def corrupt(X, sigma_occ=0.1, sigma_shift=0.1, beta=.5):
    '''
    Given marker data X as input this algorithm is used
    to randomly occlude markers (placing them at zero) or shift markers
    (adding some offset to their position).

    Args:
        X: global marker positions, dim: (n x m x 3)
        sigma_occ: variance for occlusion distribution
                   adjusts the probability of a marker being occluded
        sigma_shift: variance for shift distribution
                     adjusts the probability of a marker being shifted out of place
        beta: parameter of uniform distribution for shifting
              controls the scale of the random translations applied to shifted markers

    Returns:
        X_hat: corrupted X with same dimension
    '''
    n, m, _ = X.shape
    
    # Sample probability at which to occlude / shift markers.
    a_occ = torch.normal(0.0, sigma_occ, (n, 1), device=X.device) # (n, 1)
    a_shift = torch.normal(0.0, sigma_shift, (n, 1), device=X.device) # (n, 1)
 
    # Sample using clipped probabilities if markers are occluded / shifted.
    a_occ = torch.abs(a_occ)
    a_occ[a_occ > 2*sigma_occ] = 2*sigma_occ

    a_shift = torch.abs(a_shift)
    a_shift[a_shift > 2*sigma_shift] = 2*sigma_shift

    sampler_occ = torch.distributions.bernoulli.Bernoulli(a_occ)
    sampler_shift = torch.distributions.bernoulli.Bernoulli(a_shift)
    X_occ = sampler_occ.sample((m,)).transpose(1, 0).to(X.device) # n x m
    X_shift = sampler_shift.sample((m,)).transpose(1, 0).to(X.device) # n x m
    
    # Sample the magnitude by which to shift each marker.
    sampler_beta = torch.distributions.Uniform(low=-beta, high=beta)
    X_v = sampler_beta.sample((m, 3)).permute(2, 0, 1).to(X.device) # n x m x 3

    # Move shifted markers and place occluded markers at zero.
    X_hat = X + torch.multiply(X_v, X_shift.reshape((n, m, 1)))
    X_hat = torch.multiply(X_hat, (1 - X_occ).reshape((n, m, 1)))

    return X_hat


def preweighted_Z(w, Z):
    Z_ = torch.sum(
            torch.mul(Z.permute(1, 2, 0, 3), w[..., None, None]), 
            axis=1).permute(1, 0, 2) # n x m x 3
    return Z_


def test_lbs():
    n = 20
    j = 31
    m = 41
    w = torch.rand(m, j)
    Z = torch.rand(n, m, j, 3)
    Y = torch.rand(n, j, 3, 4)
    X = LBS(w, Y, Z)
    print(X.shape)


def test_svd():
    d = 3
    n = 20
    m = 100
    P = torch.rand(m, d, n)
    Q = torch.rand(m, d, n)
    R, t = svd_rot(P, Q)
    print(R.shape)
    print(t.shape)


def test_corrupt():
    n = 20
    m = 31
    X = torch.rand(n, m, 3)
    X_hat = corrupt(X)
    print(X_hat.shape)


if __name__ == "__main__":
    # test_lbs()
    test_svd()
    # test_corrupt()
