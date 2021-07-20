import torch
import numpy as np


main_labels =\
        ['C7', 'CLAV', 'LANK', 'LBHD', 'LBWT', 'LELB', 'LFHD', 'LFIN', 
        'LFRM', 'LFWT', 'LHEE', 'LKNE', 'LMT5', 'LSHN', 'LSHO', 'LTHI', 
        'LTOE', 'LUPA', 'LWRA', 'LWRB', 'RANK', 'RBAC', 'RBHD', 'RBWT', 
        'RELB', 'RFHD', 'RFIN', 'RFRM', 'RFWT', 'RHEE', 'RKNE', 'RMT5', 
        'RSHN', 'RSHO', 'RTHI', 'RTOE', 'RUPA', 'RWRA', 'RWRB', 'STRN', 'T10']


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


def get_Z(X, Y):
    '''
    Local offset computation function

    Parameters:
        X: global marker positions, dim: (n, m, 3)
        Y: rotation + translation matrices, dim: (n, j, 3, 4)

    Return:
        Z: local offsets, dim: (n, m, j, 3)
    '''
    Y_rot = Y[..., : 3] # n x j x 3 x 3
    Y_tr = Y[..., 3: ] # n x j x 3 x 1
    Y_rot_inv = Y_rot.transpose(0, 1, 3, 2) # n x j x 3 x 3
    Y_tr_inv = -Y_tr # n x j x 3 x 1

    Y_rot_x_tr = np.matmul(Y_rot_inv, Y_tr_inv) # n x j x 3 x 1
    Y_inv = np.concatenate([Y_rot_inv, Y_rot_x_tr], axis = 3) # n x j x 3 x 4

    X_expand = np.expand_dims(X, axis = 3) # n x m x 3 x 1

    Y_inv_rot = np.expand_dims(Y_inv[..., : 3], 2) # n x m x 1 x 3 x 3
    Y_inv_tr = np.expand_dims(Y_inv[..., 3: ], 2) # n x m x 1 x 3 x 1

    Z = np.matmul(Y_inv_rot, np.expand_dims(X_expand, 1)) + Y_inv_tr
    Z = Z.transpose((0, 2, 1, 3, 4))
    Z = np.squeeze(Z)
    return Z


def clean_XYZ(X, Y, avg_bone, mm_conv = 0.056444):
    nans = np.isnan(X)[0, :, :].transpose()
    nans_float = 1 - nans.astype(float)
    nans_expand = np.expand_dims(nans_float, 2)
    nans_expand2 = np.expand_dims(nans_expand, 3)
    
    X = np.nan_to_num(X)
    X = X.transpose(2, 1, 0)
    X *= nans_expand
    X[:, :, :3] *= (1 / avg_bone)
    X = X[..., 0: 3]

    Y[..., 3] *= (1 / avg_bone) * mm_conv
    
    Z = get_Z(X, Y)
    Z *= nans_expand2

    return X, Y, Z


def LBS_np(w, Y, Z):
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


def LBS_torch(w, Y, Z):
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


def svd_rot_np(P, Q):
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
    d, n = P.shape[-2:]

    # X,Y are n x k
    P_ = np.sum(P, axis=-1) / n
    Q_ = np.sum(Q, axis=-1) / n
    X = P - P_[:, :, None]
    Y = Q - Q_[:, :, None]
    Yt = Y.transpose(0, 2, 1)

    # S is n x n
    S = X @ Yt

    # U, V are n x m
    U, _, V_t = np.linalg.svd(S)
    V = V_t.transpose(0, 2, 1)
    Ut = U.transpose(0, 2, 1)

    det = np.linalg.det(V @ Ut)
    Ut[:, -1, :] *= det.reshape((-1, 1))

    # R is n x n
    R = V @ Ut

    # t is n x k
    t = Q_.reshape((-1, d, 1)) - R @ P_.reshape((-1, d, 1))

    return R, t


def svd_rot_torch(P, Q):
    assert P.shape == Q.shape
    d, n = P.shape[-2:]

    # X,Y are n x k
    P_ = torch.sum(P, axis=-1) / n
    Q_ = torch.sum(Q, axis=-1) / n
    X = P - P_[:, :, None]
    Y = Q - Q_[:, :, None]
    Yt = Y.permute(0, 2, 1)

    # S is n x n
    S = torch.matmul(X, Yt)

    # U, V are n x m
    U, _, V = torch.svd(S)
    # V = V_t.permute(0, 2, 1)
    Ut = U.permute(0, 2, 1)

    det = torch.det(torch.matmul(V, Ut))
    Ut[:, -1, :] *= det.view(-1, 1)

    # R is n x n
    R = torch.matmul(V, Ut)

    # t is n x k
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


def corrupt_np(X, sigma_occ=0.1, sigma_shift=0.1, beta=50):
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
    X_ = LBS_np(w, Y, Z)
    X_ = torch.Tensor(X_)
    print(X_.shape)

    # w = np.ones((m, j))
    # Z = np.ones((n, m, j, 3))*15.4
    # Y = np.ones((n, j, 3, 4))*30.3

    # Z = np.arange(n*m*j*3).reshape((n, m, j, 3))
    # Y = np.arange(n*j*12).reshape((n, j, 3, 4))

    w = torch.Tensor(w)
    Z = torch.Tensor(Z)
    Y = torch.Tensor(Y)
    X = LBS_torch(w, Y, Z)
    print(X.shape)
    diff = torch.abs(X_ - X).sum().data
    print(torch.max(torch.abs(X_ - X)))
    print(diff)


def test_svd():
    d = 3
    n = 20
    m = 100

    # w = np.random.rand(m, n, 1)
    P = np.random.rand(m, d, n)
    Q = np.random.rand(m, d, n)
    R_, t_ = svd_rot_np(P, Q)
    R_ = torch.Tensor(R_)
    t_ = torch.Tensor(t_)
    print(R_.shape)
    print(t_.shape)

    # w = torch.Tensor(w)
    P = torch.Tensor(P)
    Q = torch.Tensor(Q)
    R, t = svd_rot_torch(P, Q)
    # R = R.cpu().detach().numpy()
    # t = t.cpu().detach().numpy()
    print(R.shape)
    print(t.shape)
    diff_r = torch.abs(R_ - R).sum().data
    diff_t = torch.abs(t_ - t).sum().data
    print(torch.max(torch.abs(R_ - R)))
    print(torch.max(torch.abs(t_ - t)))
    print(diff_r, diff_t)


def test_corrupt():
    n = 20
    m = 31
    X = np.random.rand(n, m, 3)
    X_hat_np = corrupt_np(X)
    X_hat_np = torch.Tensor(X_hat_np)
    print(X_hat_np.shape)
    X = torch.Tensor(X)
    X_hat = corrupt_torch(X)
    print(X_hat.shape)
    diff = torch.abs(X_hat - X_hat_np).sum().data
    print(torch.max(torch.abs(X_hat - X_hat_np)))
    print(diff)


if __name__ == "__main__":
    # test_lbs()
    test_svd()
    # test_corrupt()
