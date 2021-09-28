import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset
from tools.preprocess import clean_XY, get_Z, local_frame


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


def read_file_sy(fname):
    data = np.load(fname)
    X_read = data["raw_markers"]
    Y_read = np.concatenate([data["J_R"], data["J_t"][..., None]], axis=-1)
    Z_read = data["marker_configuration"]
    bone_len = np.linalg.norm(data["J_t_local"], axis=-1)
    avg_bone_read = bone_len.sum() / (bone_len.shape[0] - 1)

    X = torch.tensor(X_read)
    Y = torch.tensor(Y_read)
    Z = torch.tensor(Z_read)
    avg_bone = torch.tensor(avg_bone_read)
    X *= (1.0 / avg_bone)
    Y[..., 3] *= (1.0 / avg_bone)
    Z *= (1.0 / avg_bone)
    
    return (X, Y, Z, avg_bone)


class RS_Dataset(Dataset):
    def __init__(self, csv_file, file_stems, lrf_mean_markers_file, num_marker, num_joint):
        with open(file_stems) as f:
            file_stem = f.readlines()
            self.data_list = [line.strip() for line in file_stem]

        self.df = pd.read_csv(csv_file)
        self.df = self.df.loc[self.df['file_stem'].isin(self.data_list)]

        fnames = zip(self.df["marker_npy_path"].values,\
                        self.df["motion_npy_path"].values, self.df["avg_bone"].values)
        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_files, fnames)

        self.marker_data = np.concatenate([x[0] for x in data], 0)
        self.motion_data = np.concatenate([x[1] for x in data], 0)
        self.avg_bone = np.concatenate([x[2] for x in data], 0)
        self.lrf_mean_markers = torch.tensor(np.load(lrf_mean_markers_file))

    def __len__(self):
        return self.motion_data.shape[0]

    def __getitem__(self, index):
        avg_bone = self.avg_bone[index]
        X = self.marker_data[index]
        Y = self.motion_data[index]
        X = torch.tensor(X).to(torch.float32).unsqueeze(0)
        Y = torch.tensor(Y).to(torch.float32).unsqueeze(0)
        avg_bone = torch.tensor(avg_bone)

        Z = get_Z(X, Y)
        F, Y = local_frame(X, Y, self.lrf_mean_markers)
        Y, Z = Y.squeeze(0), Z.squeeze(0)
        return Y, Z, F, avg_bone


class RS_Synthetic(Dataset):
    def __init__(self, data_dir, lrf_mean_markers_file, num_marker, num_joint, local_ref_markers):
        self.local_frame_markers = local_ref_markers
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_sy, fnames)

        self.marker_data = [m[0] for m in data]
        self.motion_data = [m[1] for m in data]
        self.local_offsets = [m[2] for m in data]
        self.avg_bone = [m[3] for m in data]
        self.lrf_mean_markers = torch.tensor(np.load(lrf_mean_markers_file))

        self.indices = []
        for f_idx in range(len(self.marker_data)):
            sublist = [(f_idx, pivot) for pivot in range(self.marker_data[f_idx].shape[0])]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        avg_bone = self.avg_bone[f_idx]
        X = self.marker_data[f_idx][pivot].clone()
        Y = self.motion_data[f_idx][pivot].clone()
        Z = self.local_offsets[f_idx].clone()
        X = X.to(torch.float32).unsqueeze(0)
        Y = Y.to(torch.float32).unsqueeze(0)
        Z = Z.to(torch.float32)

        F, Y = local_frame(X, Y, self.lrf_mean_markers, self.local_frame_markers)
        Y = Y.squeeze(0)
        return Y, Z, F, avg_bone


class RS_Test_Dataset(Dataset):
    def __init__(self, csv_file, file_stems, lrf_mean_markers_file, num_marker, num_joint):
        with open(file_stems) as f:
            file_stem = f.readlines()
            self.data_list = [line.strip() for line in file_stem]

        self.df = pd.read_csv(csv_file)
        self.df = self.df.loc[self.df['file_stem'].isin(self.data_list)]

        fnames = zip(self.df["marker_npy_path"].values,\
                        self.df["motion_npy_path"].values, self.df["avg_bone"].values)
        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_files, fnames)

        self.marker_data = np.concatenate([x[0] for x in data], 0)
        self.motion_data = np.concatenate([x[1] for x in data], 0)
        self.avg_bone = np.concatenate([x[2] for x in data], 0)
        self.lrf_mean_markers = torch.tensor(np.load(lrf_mean_markers_file))

        self.marker_data = np.expand_dims(self.marker_data, 0)
        self.motion_data = np.expand_dims(self.motion_data, 0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        avg_bone = self.avg_bone[:]
        X = self.marker_data[index]
        Y = self.motion_data[index]
        X = torch.tensor(X).to(torch.float32)
        Y = torch.tensor(Y).to(torch.float32)
        avg_bone = torch.tensor(avg_bone)

        Z = get_Z(X, Y)
        F, Y = local_frame(X, Y, self.lrf_mean_markers)
        Y, Z = Y.squeeze(0), Z.squeeze(0)
        return Y, Z, F, avg_bone


class RS_Synthetic_Test(Dataset):
    def __init__(self, data_dir, lrf_mean_markers_file, num_marker, num_joint, local_ref_markers):
        self.local_frame_markers = local_ref_markers
        if isinstance(data_dir, list):
            fnames = data_dir
        else:
            with open(data_dir) as f:
                all_data = f.readlines()
                fnames = [line.strip() for line in all_data]

        with multiprocess.Pool(processes = os.cpu_count()) as pool:
            data = pool.map(read_file_sy, fnames)

        self.marker_data = [m[0] for m in data]
        self.motion_data = [m[1] for m in data]
        self.local_offsets = [m[2] for m in data]
        self.avg_bone = [m[3] for m in data]
        self.lrf_mean_markers = torch.tensor(np.load(lrf_mean_markers_file))

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, index):
        avg_bone = self.avg_bone[index]
        X = self.marker_data[index].clone()
        Y = self.motion_data[index].clone()
        Z = self.local_offsets[index].clone()
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        Z = Z.to(torch.float32)

        F, Y = local_frame(X, Y, self.lrf_mean_markers, self.local_frame_markers)
        Y = Y.squeeze(0)
        return Y, Z, F, avg_bone


def test():
    # X, Y, Z, avg_bone = read_file_sy("dataset/synthetic/01_03_poses_0.npz")
    # print(X.shape, Y.shape, Z.shape, avg_bone.shape)
    bs = 256
    import os
    from torch.utils.data import DataLoader
    fnames = [os.path.join("dataset/synthetic", path) for path in os.listdir("dataset/synthetic")]
    # dataset = RS_Synthetic(fnames[:2], "dataset/LRF_mean_offsets_synthetic.npy", 56, 24, [3, 7, 11, 21, 32, 36, 46, 54])
    # print(len(dataset))
    bs = 1
    dataset = RS_Synthetic_Test(fnames[:1], "dataset/LRF_mean_offsets_synthetic.npy", 56, 24, [3, 7, 11, 21, 32, 36, 46, 54])
    data_loader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=8, pin_memory=True)
    Y, Z, F, avg_bone = next(iter(data_loader))
    print(Y.shape, Z.shape, F.shape, avg_bone)

if __name__ == "__main__":
    test()