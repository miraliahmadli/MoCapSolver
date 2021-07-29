import torch
import numpy as np


def denormalize_Y(Y_hat, Y):
    '''
    Args:
        Y : rotation and translation matrices
            n x j x 3 x 4
    Returns:
        Y_mu : mean, j x 3 x 4
        Y_std : standard deviation, j x 3 x 4
    '''
    Y_mu = torch.mean(Y, 0).unsqueeze(0)
    Y_std = torch.std(Y, 0, unbiased=True).unsqueeze(0)
    Y_denorm = (Y_hat * Y_std) + Y_mu
    return Y_denorm


def normalize_X(X_hat, X):
    '''
    Args:
        X : rotation and translation matrices
            n x m x 3
    Calculate:
        X_mu : mean, m x 3
        X_std : standard deviation, m x 3

    Returns:
        X: normalized X
    '''
    X_mu = torch.mean(X, 0)
    X_std = torch.std(X, 0, unbiased=True)

    X_norm = (X_hat - X_mu.unsqueeze(0)) / X_std.unsqueeze(0)
    return X_norm


def normalize_Z_pw(Z):
    '''
    Args:
        Z_hat : preweighted local offsets
                n x m x 3
    Calculate:
        Z_mu : mean, m x 3
        Z_std : standard deviation, m x 3

    Returns:
        Z: normalized Z
    '''
    Z_mu = torch.mean(Z, 0)
    Z_std = torch.std(Z, 0, unbiased=True)

    Z_norm = (Z - Z_mu.unsqueeze(0)) / Z_std.unsqueeze(0)
    return Z_norm


def get_stat_Z(Z, eps=0.01):
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
    Z_ = Z_.view(n, -1)
    Z_cov = Z_.transpose(-1, -2) @ Z_
    if n != 1:
        Z_cov /= (n-1)

    # Make positive sym definite
    diag = Z_cov.diagonal()
    diag += eps

    Z_mu = Z_mu.view(-1)
    return Z_mu, Z_cov


def sample_Z(z_mu, z_cov, batch_size):
    scale_tril = torch.cholesky(z_cov)
    sampler = torch.distributions.multivariate_normal.MultivariateNormal(loc=z_mu, scale_tril=scale_tril)
    Z = sampler.sample((batch_size, ))
    return Z


def is_psd(mat):
    return bool(torch.all(torch.eig(mat)[0][:,0]>=0))


def test_stats():
    z = np.random.rand(128, 41, 31, 3)
    z_cov_np = np.cov(z.reshape((128, -1)).transpose(1, 0), bias=False)
    z_cov_np = torch.tensor(z_cov_np)
    print(is_psd(z_cov_np))
    print(z_cov_np.shape)
    z = torch.tensor(z).to("cuda")
    z_mu, z_cov = get_stat_Z(z)
    print("z_mu", z_mu.shape)
    print("z_cov", z_cov.shape)
    diag = z_cov.diagonal()
    # diag += 0.00000001
    print(is_psd(z_cov))
    z_sample = sample_Z(z_mu, z_cov, z.shape[0])
    # print(z_sample)
    print(z_sample.shape)
    

    z_cov_np = torch.tensor(z_cov_np, device="cuda")
    diff = torch.abs(z_cov - z_cov_np).sum().data
    print(torch.max(torch.abs(z_cov_np - z_cov)))
    print(diff)


if __name__ == "__main__":
    test_stats()
