import os, sys
sys.path.append("../")

import os
import numpy as np
import pandas as pd
import multiprocess

from tools.preprocess import clean_XYZ_np as clean_data
from tools.statistics import *


def read_and_get_z(csv_file = "../dataset/meta_data.csv"):
    df = pd.read_csv(csv_file)

    def read_files(inputs):
        marker_path, motion_path, avg_bone = inputs
        X_read = np.load("../" + marker_path).astype('float32')
        Y_read = np.load("../" + motion_path).astype('float32')
        if X_read.shape[-1] != Y_read.shape[0]:
            return None
        X, Y, Z = clean_data(X_read, Y_read, avg_bone)
        return Z

    fnames = zip(df["marker_npy_path"].values,\
                df["motion_npy_path"].values, df["avg_bone"].values)

    with multiprocess.Pool(processes = os.cpu_count()) as pool:
        data = pool.map(read_files, fnames)

    Z = np.concatenate([z for z in data if not (z is None)], 0)

    return Z


def calculate_mean_cov(Z):
    n = Z.shape[0]
    Z_mu = np.mean(Z, 0)
    np.save("z_stats/mean_z.npy", Z_mu)

    Z -= Zmu[None, ...]
    Z.shape = (n, -1)
    Z_cov = (1 / (n-1)) *  Z.T @ Z

    Z_cov = torch.tensor(Z_cov)
    diag = Z_cov.diagonal()
    # diag.setflags(write=True)
    while True:
        if not is_psd(Z_cov):
            diag += 0.00001
        else:
            np.save("z_stats/cov_z.npy", Z_cov.detach().cpu().numpy())
            break

    scale_tril = torch.cholesky(Z_cov)
    np.save("z_stats/cholesky_z.npy", scale_tril)


if __name__ == "__main__":
    Z = read_and_get_z()
    calculate_mean_cov(Z)
