import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset
from tools.transform import matrix_to_quaternion


def read_file_ae(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J"]
    offsets = data["marker_configuration"]
    skeleton = torch.tensor(skeleton)
    offsets = torch.tensor(offsets)

    # get motion
    R = data["J_R"]
    root_t = data["J_t"][:, 0]
    R = torch.tensor(R)
    root_t = torch.tensor(root_t)
    motion_quat = matrix_to_quaternion(R)
    motion_quat = motion_quat.view(motion_quat.shape[0], -1)
    motion_quat = torch.cat((motion_quat, root_t), dim=-1)

    return offsets, skeleton, motion_quat


def read_file_ms(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J"]
    offsets = data["marker_configuration"]
    skeleton = torch.tensor(skeleton)
    offsets = torch.tensor(offsets)

    # markers
    raw_markers = data["raw_markers"]
    raw_markers = torch.tensor(raw_markers)

    # get motion
    R = data["J_R"] # num_frames x j x 3 x 3
    root_t = data["J_t"][:, 0] # num_frames x 3
    R = torch.tensor(R)
    root_t = torch.tensor(root_t)
    motion_quat = matrix_to_quaternion(R) # num_frames x j x 4
    motion_quat = motion_quat.view(motion_quat.shape[0], -1)
    motion_quat = torch.cat((motion_quat, root_t), dim=-1)

    return offsets, skeleton, motion_quat, raw_markers


class AE_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64):
        self.window_size = window_size
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_ae, fnames)

        self.X_c = [m[0] for m in data]
        self.X_t = [m[1] for m in data]
        self.X_m = [m[2] for m in data]

        self.indices = []
        for f_idx in range(len(self.X_m)):
            sublist = [(f_idx, pivot) for pivot in range(self.X_m[f_idx].shape[0] - self.window_size)]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        X_c = self.X_c[f_idx] # m x j x 3
        X_t = self.X_t[f_idx] # j x 3
        X_m = self.X_m[f_idx][pivot : pivot + self.window_size] # T x (J*4 + 3)

        return X_c, X_t, X_m


class MS_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64):
        self.window_size = window_size
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_ms, fnames)

        self.X_c = [m[0] for m in data]
        self.X_t = [m[1] for m in data]
        self.X_m = [m[2] for m in data]
        self.raw = [m[3] for m in data]

        self.indices = []
        for f_idx in range(len(self.X_m)):
            sublist = [(f_idx, pivot) for pivot in range(self.X_m[f_idx].shape[0] - self.window_size)]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        X_c = self.X_c[f_idx] # m x j x 3
        X_t = self.X_t[f_idx] # j x 3
        X_m = self.X_m[f_idx][pivot : pivot + self.window_size] # T x (J*4 + 3)
        raw_markers = self.raw[f_idx][pivot : pivot + self.window_size] # T x m x 3

        # TODO: normalize raw_markers using reference frame
        ref_frame = raw_markers[0]

        return X_c, X_t, X_m


def test():
    bs = 256
    import os
    from torch.utils.data import DataLoader
    fnames = [os.path.join("dataset/synthetic", path) for path in os.listdir("dataset/synthetic")]
    # offsets, skeleton, motion_quat, raw_markers = read_file_ms(fnames[0])

    # print(offsets.shape)
    # print(skeleton.shape)
    # print(motion_quat.shape)
    # print(raw_markers.shape)
    # return
    print(fnames[0])
    dataset = MS_Dataset(fnames[:2])
    print(len(dataset))
    
    data_loader = DataLoader(dataset, batch_size=bs,
                            shuffle=False, num_workers=8, pin_memory=True)
    X_c, X_t, X_m = next(iter(data_loader)) 
    print(X_c.shape)
    print(X_t.shape)
    print(X_m.shape)
    # print(norm_markers.shape)


if __name__ == "__main__":
    test()