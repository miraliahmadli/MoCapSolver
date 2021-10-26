import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset
from tools.transform import matrix_to_quaternion, quaternion_to_matrix
from tools.utils import xform_inv, xform_to_mat44
from tools.preprocess import local_frame_F


def read_file_ts(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J_t_local"]
    skeleton = torch.tensor(skeleton)[None, :]
    return skeleton


class TS_Dataset(Dataset):
    def __init__(self, data_dir):
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_ts, fnames)

        self.X_t = torch.cat(data, 0)

    def __len__(self):
        return self.X_t.shape[0]

    def __getitem__(self, index):
        X_t = self.X_t[index].clone() # j x 3
        return X_t


def read_file_mc(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J_t_local"]
    skeleton = torch.tensor(skeleton)[None, :]

    offsets = data["marker_configuration"]
    offsets = torch.tensor(offsets)[None, :]
    return offsets, skeleton


class MC_Dataset(Dataset):
    def __init__(self, data_dir):
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_mc, fnames)

        self.X_c = torch.cat([m[0] for m in data], 0)
        self.X_t = torch.cat([m[1] for m in data], 0)

    def __len__(self):
        return self.X_c.shape[0]

    def __getitem__(self, index):
        X_c = self.X_c[index].clone() # m x j x 3
        X_t = self.X_t[index].clone() # j x 3
        return X_c, X_t


def read_file_motion(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J_t_local"]
    skeleton = torch.tensor(skeleton)

    # get motion
    R = data["J_R_local"]
    root_t = data["J_t"][:, 0]
    R = torch.tensor(R)
    root_t = torch.tensor(root_t)
    motion_quat = matrix_to_quaternion(R)
    motion_quat = motion_quat.view(motion_quat.shape[0], -1)
    motion_quat = torch.cat((motion_quat, root_t), dim=-1)

    return skeleton, motion_quat


class Motion_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64, overlap=32):
        self.window_size = window_size
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_motion, fnames)

        self.X_t = [m[0] for m in data]
        self.X_m = [m[1] for m in data]

        self.indices = []
        for f_idx in range(len(self.X_m)):
            sublist = [(f_idx, pivot) for pivot in range(0, self.X_m[f_idx].shape[0] - self.window_size, self.window_size - overlap)]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        X_t = self.X_t[f_idx].clone() # j x 3
        X_m = self.X_m[f_idx][pivot : pivot + self.window_size].clone() # T x (J*4 + 3)
        return X_t, X_m


def test():
    bs = 256
    import os
    from torch.utils.data import DataLoader
    fnames = [os.path.join("dataset/syn_new_local/train_sample_data", path) for path in os.listdir("dataset/syn_new_local/train_sample_data")]

    # dataset = TS_Dataset(fnames[30: 33])
    # dataset = MC_Dataset(fnames[30:33])
    dataset = Motion_Dataset(fnames[30:33])
    data_loader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=8, pin_memory=True)
    # X_t = next(iter(data_loader)) 
    # print(X_t.shape)

    # X_c, X_t = next(iter(data_loader)) 
    # print(X_c.shape)
    # print(X_t.shape)

    X_t, X_m = next(iter(data_loader)) 
    print(X_t.shape)
    print(X_m.shape)

if __name__ == "__main__":
    test()
