import torch
import numpy as np


def get_stat_Y(Y):
    '''
    Args:
        Y : rotation and translation matrices
            n x j x 3 x 4
    Returns:
        Y_mu : mean, j x 3 x 4
        Y_std : standard deviation, j x 3 x 4
    '''
    Y_mu = torch.mean(Y, 0)
    Y_std = torch.std(Y, 0, unbiased=True)
    return Y_mu, Y_std


def get_stat_X(X):
    '''
    Args:
        X : rotation and translation matrices
            n x m x 3
    Returns:
        X_mu : mean, m x 3
        X_std : standard deviation, m x 3
    '''
    X_mu = torch.mean(X, 0)
    X_std = torch.std(X, 0, unbiased=True)
    return X_mu, X_std


def get_stat_Z_preweighted(Z):
    '''
    Args:
        Z_hat : preweighted local offsets
                n x m x 3
    Returns:
        Z_mu : mean, m x 3
        Z_std : standard deviation, m x 3
    '''
    Z_mu = torch.mean(Z, 0)
    Z_std = torch.std(Z, 0, unbiased=True)
    return Z_mu, Z_std


def get_stat_Z(Z):
    '''
    Args:
        Z : local offsets
            n x m x j x 3
    Returns:
        Z_mu : mean, m x j x 3
        Z_cov : covariance matrix, (m x j x 3) x (m x j x 3)
    '''
    n = Z.shape[0]
    Z_mu = torch.mean(Z, 0)

    Z_ = Z - Z_mu[None, ...]
    Z_ = Z_.view(n, -1).transpose(-1, -2)
    Z_cov = 1 / (n-1) * (Z_ @ Z_.transpose(-1, -2))
    return Z_mu, Z_cov


def test_stats():
    z = np.random.rand(100, 41, 31, 3)
    z_cov_np = np.cov(z.reshape((100, -1)).transpose(1, 0), bias=False)
    z = torch.tensor(z)
    z_mu, z_cov = get_stat_Z(z)
    print("z_mu", z_mu.shape)
    print("z_cov", z_cov.shape)

    z_cov_np = torch.tensor(z_cov_np)
    diff = torch.abs(z_cov - z_cov_np).sum().data
    print(torch.max(torch.abs(z_cov_np - z_cov)))
    print(diff)


if __name__ == "__main__":
    test_stats()
