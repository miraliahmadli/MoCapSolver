import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset

local_frame_markers = [3, 7, 11, 21, 32, 36, 46, 54]

def read_file(fname):
    data = np.load(fname)
    clean = data["clean_markers"]
    raw = data["raw_markers"]
    clean = torch.tensor(clean)
    raw = torch.tensor(raw)
    return clean, raw


class MR_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64, threshold=0.8):
        self.window_size = window_size
        self.thr = threshold
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file, fnames)

        self.clean_markers = [m[0] for m in data]
        self.raw_markers = [m[1] for m in data]

        self.indices = []
        for f_idx in range(len(self.clean_markers)):
            sublist = [(f_idx, pivot) for pivot in range(self.clean_markers[f_idx].shape[0] - self.window_size)]
            self.indices += sublist

    def get_gt_scores(self, raw, clean):
        distance = torch.norm(raw - clean, 2, dim=-1)
        rel_score = torch.maximum(torch.tensor(0), torch.minimum(torch.tensor(1), 1.2 - 0.4*distance))
        rel_score = (rel_score >= self.thr).to(torch.float32)
        return rel_score

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        clean = self.clean_markers[f_idx][pivot : pivot + self.window_size] # T x m x 3
        raw = self.raw_markers[f_idx][pivot : pivot + self.window_size] # T x m x 3
        ref_clean = clean[:, local_frame_markers] # T x 8 x 3
        ref_raw = raw[:, local_frame_markers] # T x 8 x 3
        gt_rel = self.get_gt_scores(ref_raw, ref_clean)
        return raw, gt_rel


def test():
    bs = 256
    import os
    from torch.utils.data import DataLoader
    fnames = [os.path.join("dataset/synthetic", path) for path in os.listdir("dataset/synthetic")]
    print(fnames[0])
    dataset = MR_Dataset(fnames[:2])
    print(len(dataset))
    train_steps = len(dataset) // bs
    
    data_loader = DataLoader(dataset, batch_size=bs,\
                                shuffle=False, num_workers=8, pin_memory=True)
    raw, gt = next(iter(data_loader)) 
    print(raw.shape)
    print(gt.shape)
    print(gt[0, 10:15])
    print(gt[gt == 0])
    print(raw[0, 10:15, local_frame_markers])


if __name__ == "__main__":
    test()
