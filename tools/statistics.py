import torch
import numpy as np


def get_stat_Y(Y):
    '''
    Parameters:
        Y : rotation and translation matrices
            n x j x 3 x 4
    Returns:
        Y_mu : mean, j x 3 x 4
        Y_std : standard deviation, j x 3 x 4
    '''
    Y_mu = torch.mean(Y, 0)
    Y_std = torch.std(Y, 0, unbiased=False)
    return Y_mu, Y_std


def get_stat_X(X):
    '''
    Parameters:
        X : rotation and translation matrices
            n x m x 3
    Returns:
        X_mu : mean, m x 3
        X_std : standard deviation, m x 3
    '''
    X_mu = torch.mean(X, 0)
    X_std = torch.std(X, 0, unbiased=False)
    return X_mu, X_std


def get_stat_Z_preweighted(Z):
    '''
    Parameters:
        Z : preweighted local offsets
            n x m x 3
    Returns:
        Z_mu : mean, m x 3
        Z_std : standard deviation, m x 3
    '''
    Z_mu = torch.mean(Z, 0)
    Z_std = torch.std(Z, 0, unbiased=False)
    return Z_mu, Z_std


def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=0).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)


def get_stat_Z(Z):
    '''
    Parameters:
        Z : local offsets
            n x m x j x 3
    Returns:
        Z_mu : mean, m x j x 3
        Z_cov : covariance matrix, (m x j x 3) x (m x j x 3)
    '''
    n = Z.shape[0]
    Z_mu = torch.mean(Z, 0)

    Z_ = Z - Z_mu[None, :, :, :]
    Z_ = Z_.view(n, -1)
    Z_cov = 1 / (n-1) * (Z_ @ Z_.transpose(-1, -2))
    return Z_mu, Z_cov


def test_stats():
    pass
