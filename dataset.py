import numpy as np

import torch
from torch.utils.data import Dataset

class MoCap(Dataset):
    def __init__(self, data_dir, fnames, weights):
        with open(fnames) as f:
            self.data_list = [line.split(',') for line in f]
        self.data_dir = data_dir
        self.weights = weights

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path_X, path_Y = self.data_list[index]
        X_global = get_X(path_X)
        Y = get_Y(path_Y)
        Z = get_Z(X_global, Y)
        X = LBS(self.weights, Y, Z)


def collate_fn(batch):
    pass
