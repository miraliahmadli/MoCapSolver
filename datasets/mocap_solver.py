import os
import numpy as np
import pandas as pd
import multiprocess

import torch
from torch.utils.data import Dataset
from tools.transform import matrix_to_quaternion, quaternion_to_matrix
from tools.utils import xform_inv, xform_to_mat44
from tools.preprocess import local_frame_F


def read_file_ae(fname):
    data = np.load(fname)
    # T-pose
    skeleton = data["J_t_local"]
    offsets = data["marker_configuration"]
    skeleton = torch.tensor(skeleton)
    offsets = torch.tensor(offsets)

    # get motion
    R = data["J_R_local"]
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
    skeleton = data["J_t_local"]
    offsets = data["marker_configuration"]
    skeleton = torch.tensor(skeleton, dtype=torch.float32)
    offsets = torch.tensor(offsets, dtype=torch.float32)

    # markers
    raw_markers = data["raw_markers"]
    raw_markers = torch.tensor(raw_markers, dtype=torch.float32)
    clean_markers = data["clean_markers"]
    clean_markers = torch.tensor(clean_markers, dtype=torch.float32)

    # get motion
    R = data["J_R_local"] # num_frames x j x 3 x 3
    root_t = data["J_t"][:, 0] # num_frames x 3
    R = torch.tensor(R, dtype=torch.float32)
    root_t = torch.tensor(root_t, dtype=torch.float32)
    motion_quat = matrix_to_quaternion(R) # num_frames x j x 4
    motion_quat = motion_quat.view(motion_quat.shape[0], -1)
    motion_quat = torch.cat((motion_quat, root_t), dim=-1)
    global_xform = torch.tensor(np.concatenate([data["J_R"], data["J_t"][..., None]], axis=-1), dtype=torch.float32)

    return offsets, skeleton, motion_quat, raw_markers, global_xform, clean_markers


class AE_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64, overlap=32):
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
            sublist = [(f_idx, pivot) for pivot in range(0, self.X_m[f_idx].shape[0] - self.window_size, self.window_size - overlap)]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        X_c = self.X_c[f_idx].clone() # m x j x 3
        X_t = self.X_t[f_idx].clone() # j x 3
        X_m = self.X_m[f_idx][pivot : pivot + self.window_size].clone() # T x (J*4 + 3)

        return X_c, X_t, X_m


class MS_Dataset(Dataset):
    def __init__(self, data_dir, window_size=64, overlap=32, local_ref_markers=[3, 7, 11, 21, 32, 36, 46, 54], local_ref_joint=3):
        self.window_size = window_size
        self.local_ref_markers = local_ref_markers
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
        self.clean = [m[5] for m in data]

        xform_ref_joint = [m[4][:, local_ref_joint] for m in data]
        xform_inv_ref_joint = [xform_inv(m[4][:, local_ref_joint]).unsqueeze(1) for m in data]
        self.ref_marker_pos = [(xform_inv_ref_joint[i][...,:3] @ m[3][:, local_ref_markers, :, None] + xform_inv_ref_joint[i][...,3, None]).squeeze(-1) for i, m in enumerate(data)]

        self.indices = []
        for f_idx in range(len(self.X_m)):
            sublist = [(f_idx, pivot) for pivot in range(0, self.X_m[f_idx].shape[0] - self.window_size, self.window_size - overlap)]
            self.indices += sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        f_idx, pivot = self.indices[index]
        X_c = self.X_c[f_idx] # m x j x 3
        X_t = self.X_t[f_idx] # j x 3
        X_m = self.X_m[f_idx][pivot : pivot + self.window_size].clone() # T x (J*4 + 3)
        raw_markers = self.raw[f_idx][pivot : pivot + self.window_size].clone() # T x m x 3
        clean_markers = self.clean[f_idx][pivot : pivot + self.window_size].clone()
        ref_marker_pos = self.ref_marker_pos[f_idx][pivot]

        F = local_frame_F(raw_markers, ref_marker_pos.unsqueeze(1), self.local_ref_markers)
        F_inv = xform_inv(F)

        root_rot = quaternion_to_matrix(X_m[..., :4])
        root_t = X_m[..., -3:].unsqueeze(-1)
        root_xform = torch.cat((root_rot, root_t), -1)
        root_44 = xform_to_mat44(root_xform) # T x 4 x 4
        root = F_inv @ root_44 # T x 3 x 4
        root_quat = matrix_to_quaternion(root[..., :3]) # T x 4
        X_m[..., :4] = root_quat
        X_m[..., -3:] = root[..., -1]

        # from models.mocap_solver import FK
        # from models.mocap_solver.skeleton import get_topology
        # from tools.viz import visualize
        # joint_topology = get_topology("dataset/hierarchy_synthetic_bfs.txt", 24)
        # j_rot, j_tr = FK(joint_topology, X_m.unsqueeze(0), X_t.unsqueeze(0))
        # j_xform = torch.cat([j_rot, j_tr.unsqueeze(-1)], dim=-1)

        # visualize(Ys=10*j_xform[0].detach().cpu().numpy()[..., [0, 2, 1], :])
        # exit()

        nans = ~((raw_markers != 0.0).any(axis=-1))
        raw_markers_normalized = F_inv[..., :3].unsqueeze(1) @ raw_markers[..., None] + F_inv[..., 3, None].unsqueeze(1)
        raw_markers_normalized[nans] = torch.zeros((3,1)).to(torch.float32)
        clean_markers_norm = F_inv[..., :3].unsqueeze(1) @ clean_markers[..., None] + F_inv[..., 3, None].unsqueeze(1)
        return X_c, X_t, X_m, raw_markers_normalized.squeeze(-1), clean_markers_norm.squeeze(-1), F#, root_quat, root[..., -1]


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
    # print(fnames[0])
    dataset = MS_Dataset(fnames[30:32])
    # print(len(dataset))
    
    data_loader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=8, pin_memory=True)
    X_c, X_t, X_m, norm_markers, F = next(iter(data_loader)) 
    # print(X_c.shape)
    # print(X_t.shape)
    # print(X_m.shape)
    # print(norm_markers.shape)
    # print(F.shape)
    root_quat = X_m[..., :4]
    root_quat_to_mat = quaternion_to_matrix(root_quat)
    root_tr = X_m[..., -3: , None]

    # print(root_quat_to_mat.shape)
    # print(root_tr.shape)
    from tools.viz import visualize
    visualize(Xs=norm_markers[0].cpu().numpy()[None, ...] * 10.0)


if __name__ == "__main__":
    test()