import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset
from tools.preprocess import clean_XY, get_Z, local_frame


class MoCap(Dataset):
    def __init__(self, csv_file, fnames, lrf_mean_markers_file, num_marker, num_joint, test=False):
        self.is_test = test
        with open(fnames) as f:
            file_stem = f.readlines()
            self.data_list = [line.strip() for line in file_stem]

        self.df = pd.read_csv(csv_file)
        self.df = self.df.loc[self.df['file_stem'].isin(self.data_list)]

        def read_files(inputs):
            marker_path, motion_path, avg_bone = inputs
            X_read = np.load(marker_path)
            Y_read = np.load(motion_path)
            avg_bone_read = np.broadcast_to(avg_bone, (Y_read.shape[0], 1))
            X_read = torch.tensor(X_read)
            Y_read = torch.tensor(Y_read)
            avg_bone_read = torch.tensor(avg_bone_read)
            X, Y, avg_bone = clean_XY(X_read, Y_read, avg_bone_read)
            assert X.shape[0] == Y.shape[0]
            assert X.shape[0] == avg_bone.shape[0]
            return (X, Y, avg_bone)

        fnames = zip(self.df["marker_npy_path"].values,\
                        self.df["motion_npy_path"].values, self.df["avg_bone"].values)
        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_files, fnames)

        self.marker_data = np.concatenate([x[0] for x in data], 0)
        self.motion_data = np.concatenate([x[1] for x in data], 0)
        self.avg_bone = np.concatenate([x[2] for x in data], 0)
        self.lrf_mean_markers = torch.tensor(np.load(lrf_mean_markers_file))

        if test:
            self.marker_data = np.expand_dims(self.marker_data, 0)
            self.motion_data = np.expand_dims(self.motion_data, 0)

    def __len__(self):
        if self.is_test:
            return len(self.data_list)
        return self.motion_data.shape[0]

    def __getitem__(self, index):
        if self.is_test:
            avg_bone = self.avg_bone[:]
        else:
            avg_bone = self.avg_bone[index]
        X = self.marker_data[index]
        Y = self.motion_data[index]
        X = torch.tensor(X).to(torch.float32)
        Y = torch.tensor(Y).to(torch.float32)
        if not self.is_test:
            X = X.unsqueeze(0)
            Y = Y.unsqueeze(0)
        avg_bone = torch.tensor(avg_bone)

        Z = get_Z(X, Y)
        F, Y = local_frame(X, Y, self.lrf_mean_markers, "cpu")
        Y, Z = Y.squeeze(0), Z.squeeze(0)
        return Y, Z, F, avg_bone
