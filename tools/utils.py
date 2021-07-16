import torch
import numpy as np


def LBS_np(w, Z, Y):
    '''
    Linear Blend Skinning function

    Parameters:
        w: weights associated with marker offsets, dim: (m, j)
        Z: local offsets, dim: (n, m, j, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        X: global marker positions, dim: (n, m, 3)
    '''
    m, j = w.shape
    n = Z.shape[0]

    w_ = w.transpose(1, 0) # j x m

    z_padding = np.ones((n, m, j, 1))
    Z_ = np.concatenate((Z, z_padding), axis=3)
    Z_ = Z_.transpose(2, 0, 3, 1) # j x n x 4 x m

    Y_ = Y.transpose(1, 0, 2, 3) # j x n x 3 x 4

    prod = np.matmul(Y_, Z_).transpose(0, 3, 1, 2) # j x m x n x 3

    X = np.sum(
            np.multiply(prod, w_.reshape((j, m, 1, 1))), 
            axis=0).transpose(1, 0, 2) # n x m x 3

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


def svd_rot_np(P, Q, w):
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
    assert P.shape == Q.shape
    n, k = P.shape
    assert k == w.shape[0]

    # X,Y are n x k
    P_ = np.dot(P, w) / np.sum(w)
    Q_ = np.dot(P, w) / np.sum(w)
    X = P - P_
    Y = Q - Q_

    # S is n x n
    W = np.diag(w.reshape(-1))
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


def svd_rot_torch(P, Q, w):
    assert P.shape == Q.shape
    n, k = P.shape
    assert k == w.shape[0]

    # X,Y are n x k
    P_ = torch.mm(P, w) / torch.sum(w)
    Q_ = torch.mm(P, w) / torch.sum(w)
    X = P - P_
    Y = Q - Q_

    # S is n x n
    W = torch.diag(w.reshape(-1))
    S = torch.matmul(np.matmul(X, W), Y.T)

    # U, V are n x m
    U, _, V_t = torch.svd(S)
    V = V_t.T

    m = U.shape[1]
    D = torch.diag(torch.ones(m))
    D[-1, -1] = torch.det(V*U.T)

    # R is n x n
    R = torch.matmul(torch.matmul(V, D), U.T)

    # t is n x k
    t = Q_ - torch.matmul(R, P_)

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


def corrupt_np(X, sigma_occ=0.1, sigma_shift=0.1, beta=50):
    '''
    Given marker data X as input this algorithm is used
    to randomly occlude markers (placing them at zero) or shift markers
    (adding some offset to their position).

    Parameters:
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
    a_occ = np.random.normal(0, sigma_occ, n) # (n, )
    a_shift = np.random.normal(0, sigma_shift, n) # (n, )
 
    # Sample using clipped probabilities if markers are occluded / shifted.
    a_occ = np.abs(a_occ)
    a_occ[a_occ > 2*sigma_occ] = 2*sigma_occ

    a_shift = np.abs(a_shift)
    a_shift[a_shift > 2*sigma_shift] = 2*sigma_shift

    X_occ = np.array([np.random.binomial(1, occ, size=m) for occ in a_occ]) # n x m
    X_shift = np.array([np.random.binomial(1, shift, size=m) for shift in a_shift]) # n x m
    
    # Sample the magnitude by which to shift each marker.
    X_v = np.random.uniform(-beta, beta, (n, m, 3)) # n x m x 3

    # Move shifted markers and place occluded markers at zero.
    X_hat = X + np.multiply(X_v, X_shift.reshape((n, m, 1)))
    X_hat = np.multiply(X_hat, (1 - X_occ).reshape((n, m, 1)))

    return X_hat


def corrupt_torch(X, sigma_occ=0.1, sigma_shift=0.1, beta=50):
    n, m, _ = X.shape
    
    # Sample probability at which to occlude / shift markers.
    a_occ = torch.normal(0.0, sigma_occ, (n, 1)) # (n, 1)
    a_shift = torch.normal(0.0, sigma_shift, (n, 1)) # (n, 1)
 
    # Sample using clipped probabilities if markers are occluded / shifted.
    a_occ = torch.abs(a_occ)
    a_occ[a_occ > 2*sigma_occ] = 2*sigma_occ

    a_shift = torch.abs(a_shift)
    a_shift[a_shift > 2*sigma_shift] = 2*sigma_shift

    sampler_occ = torch.distributions.bernoulli.Bernoulli(a_occ)
    sampler_shift = torch.distributions.bernoulli.Bernoulli(a_shift)
    X_occ = sampler_occ.sample((m,)).transpose(1, 0) # n x m
    X_shift = sampler_shift.sample((m,)).transpose(1, 0) # n x m
    
    # Sample the magnitude by which to shift each marker.
    sampler_beta = torch.distributions.Uniform(low=-beta, high=beta)
    X_v = sampler_beta.sample((n, m, 3)) # n x m x 3

    # Move shifted markers and place occluded markers at zero.
    X_hat = X + torch.multiply(X_v, X_shift.reshape((n, m, 1)))
    X_hat = torch.multiply(X_hat, (1 - X_occ).reshape((n, m, 1)))

    return X_hat


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
    n = 20
    k = 31

    w = np.random.rand(k, 1)
    P = np.random.rand(n, k)
    Q = np.random.rand(n, k)

    w = torch.Tensor(w)
    P = torch.Tensor(P)
    Q = torch.Tensor(Q)
    # R, t = svd_rot_np(P, Q, w)
    R, t = svd_rot_torch(P, Q, w)
    print(R.shape)
    print(t.shape)


def test_corrupt():
    n = 20
    m = 31
    X = np.random.rand(n, m, 3)
    # X_hat = corrupt_np(X)
    X = torch.Tensor(X)
    X_hat = corrupt_torch(X)
    print(X_hat.shape)


if __name__ == "__main__":
    test_lbs()
    # test_svd()
