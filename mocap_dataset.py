import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tools.preprocess import clean_XYZ_torch as clean_data
from tools.preprocess import local_frame_torch as local_frame

class MoCap(Dataset):
    def __init__(self, csv_file, fnames, num_marker, num_joint, test=False):
        self.is_test = test
        with open(fnames) as f:
            file_stem = f.readlines()
            self.data_list = [line.strip() for line in file_stem]

        self.df = pd.read_csv(csv_file)
        self.df = self.df.loc[self.df['file_stem'].isin(self.data_list)]
        self.marker_data = np.empty((4, num_marker, 0))
        self.motion_data = np.empty((0, num_joint, 3, 4))
        self.avg_bone = np.empty((0, 1))

        for marker_path, motion_path, avg_bone in zip(self.df["marker_npy_path"].values,\
                            self.df["motion_npy_path"].values, self.df["avg_bone"].values):
            X_read = np.load(marker_path)
            Y_read = np.load(motion_path)
            self.marker_data = np.concatenate([self.marker_data, X_read], -1)
            self.motion_data = np.concatenate([self.motion_data, Y_read], 0)
            avg_bone = np.broadcast_to(avg_bone, (Y_read.shape[0], 1))
            self.avg_bone = np.concatenate([self.avg_bone, avg_bone], 0)

        if test:
            self.marker_data = np.expand_dims(self.marker_data, 0)
            self.motion_data = np.expand_dims(self.motion_data, 0)

    def __len__(self):
        if self.is_test:
            return len(self.data_list)
        return self.motion_data.shape[0]

    def __getitem__(self, index):
        if self.is_test:
            X_read = self.marker_data[index]
        else:
            X_read = self.marker_data[..., index]
        Y_read = self.motion_data[index]
        avg_bone = self.avg_bone[index]
        X_read = torch.tensor(X_read)
        Y_read = torch.tensor(Y_read)
        if not self.is_test:
            X_read = X_read.unsqueeze(-1)
            Y_read = Y_read.unsqueeze(0)
        avg_bone = torch.tensor(avg_bone)

        X, Y, Z = clean_data(X_read, Y_read, avg_bone)
        F, Y = local_frame(X, Y, "cpu")
        Y, Z = Y.squeeze(0), Z.squeeze(0)
        return Y, Z, F
