import numpy as np

import torch
from torch.utils.data import Dataset

class MoCap(Dataset):
    def __init__(self, data_dir, fnames):
        with open(fnames) as f:
            self.data_list = [line.split(',') for line in f]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pass


def collate_fn(batch):
    pass
