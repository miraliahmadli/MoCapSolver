import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import AE, MocapSolver, MS_loss
from models.mocap_solver import Decoder, MSD
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import MS_Dataset
from tools.transform import quaternion_to_matrix, matrix_to_axis_angle


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(MS_Agent, self).__init__(cfg, test, sweep)
        torch.autograd.set_detect_anomaly(True)
        self.train_decoder = cfg.model.train_decoder
        # self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device)
        self.joint_weights = torch.tensor([1]*self.num_joints, dtype=torch.float32, device=self.device)
        self.marker_weights = torch.tensor([1]*self.num_markers, dtype=torch.float32, device=self.device)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.alphas = cfg.loss.alphas

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        # self.auto_encoder = AE(self.edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
        #                        offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])#.to(self.device)
        # self.mocap_solver = MocapSolver(self.num_markers, self.cfg.window_size, 1024,
        #                         use_motion=True, use_marker_conf=True, use_skeleton=True)#.to(self.device)
        # self.ms_decoder = self.auto_encoder.decoder.to(self.device)
        # self.ms_decoder = Decoder(self.auto_encoder.encoder)#.to(self.device)
        self.model = MSD(self.edges).to(self.device)

        # if not self.train_decoder:
        #     if os.path.exists(self.cfg.model.decoder_dir):
        #         self.load_decoder(self.cfg.model.decoder_dir)
        #     else:
        #         print("No pretrained encoder")
        #         exit()

    def load_data(self):
        self.train_dataset = MS_Dataset(data_dir=self.cfg.train_filenames, window_size=self.cfg.window_size, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)
        self.val_dataset = MS_Dataset(data_dir=self.cfg.val_filenames, window_size=self.cfg.window_size, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, X):
        # l_c, l_t, l_m = self.model(X)
        # l_m = l_m.view(l_m.shape[0], 16, -1)
        # Y_c, Y_t, Y_m = self.ms_decoder(l_c, l_t, l_m)
        # Y_c = Y_c.view(Y_c.shape[0], self.num_markers, self.num_joints, 3)
        # Y_t = Y_t.view(Y_t.shape[0], self.num_joints, 3)
        Y_c, Y_t, Y_m = self.model(X)

        # TODO: apply skinning to get Y
        # 1. quat to matrix
        # 2. get affine matrix
        # 3. LBS(w, Y, Y_c)
        Y = X

        return Y_c, Y_t, Y_m, Y

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        total_angle_diff = 0
        total_jp_err = 0
        n = 0

        self.model.train()
        # if self.train_decoder:
        #     self.ms_decoder.train()
        for batch_idx, (X_c, X_t, X_m, X) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X = X.to(torch.float32).to(self.device)
            X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

            Y_c, Y_t, Y_m, Y = self.run_batch(X)
            losses = self.criterion((X, X_c, X_t, X_m), (Y, Y_c, Y_t, Y_m))
            loss, loss_marker, loss_c, loss_t, loss_m = losses
            total_loss += loss.item()
            total_loss_c += loss_c.item()
            total_loss_t += loss_t.item()
            total_loss_m += loss_m.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            quat_X = X_m[...,:-3].view(bs, self.cfg.window_size, self.num_joints, 4)
            quat_Y = Y_m[...,:-3].view(bs, self.cfg.window_size, self.num_joints, 4)
            R_x = quaternion_to_matrix(quat_X)
            R_y = quaternion_to_matrix(quat_Y)
            R = R_y.transpose(-2, -1) @ R_x
            axis_angle = matrix_to_axis_angle(R)
            angle_diff = axis_angle[..., -1].view(-1, self.num_joints) / np.pi * 180 # (n x t) x j
            total_angle_diff += torch.sum(torch.abs(angle_diff)) / (self.cfg.window_size * self.num_joints)

            jp_err = torch.norm(X_t - Y_t, 2, dim=-1) # n x j
            total_jp_err += jp_err.sum()
            # mp_err = 

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, angle_diff={2:.4f}, jpe={3:.4f}".format(epoch, loss.item() / bs, angle_diff.mean(), jp_err.mean())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_loss_c /= n
        total_loss_t /= n
        total_loss_m /= n
        total_angle_diff *= 180 / (n * np.pi)
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        # tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}, angle_diff={2:.4f}".format(epoch, total_loss, total_angle_diff.mean())
        # tqdm_batch.set_postfix_str(tqdm_update)
        # tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_diff = 0
        total_jp_err = 0
        n = 0

        self.model.eval()
        # if self.train_decoder:
        #     self.ms_decoder.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m, X) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X = X.to(torch.float32).to(self.device)
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

                Y_c, Y_t, Y_m, Y = self.run_batch(X)
                losses = self.criterion((X, X_c, X_t, X_m), (Y, Y_c, Y_t, Y_m))
                loss, loss_marker, loss_c, loss_t, loss_m = losses
                total_loss += loss.item()

                quat_X = X_m[...,:-3].view(bs, self.cfg.window_size, self.num_joints, 4)
                quat_Y = Y_m[...,:-3].view(bs, self.cfg.window_size, self.num_joints, 4)
                R_x = quaternion_to_matrix(quat_X)
                R_y = quaternion_to_matrix(quat_Y)
                R = R_y.transpose(-2, -1) @ R_x
                axis_angle = matrix_to_axis_angle(R)
                angle_diff = axis_angle[..., -1].view(-1, self.num_joints) / np.pi * 180 # (n x t) x j
                total_angle_diff += torch.sum(torch.abs(angle_diff)) / (self.cfg.window_size * self.num_joints)

                jp_err = torch.norm(X_t - Y_t, 2, dim=-1) # n x j
                total_jp_err += jp_err.sum()

                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, angle_diff={2:.4f}, jpe={3:.4f}".format(epoch, loss.item() / bs, angle_diff.mean(), jp_err.mean())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_angle_diff *= 180 / (n * np.pi)
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}, angle_diff={2:.4f}".format(epoch, total_loss, total_angle_diff.mean())
        # tqdm_batch.set_postfix_str(tqdm_update)
        # tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def test_one_animation(self):
        pass

    def build_loss_function(self):
        return MS_loss(self.joint_weights, self.marker_weights, self.alphas)

    def load_decoder(self, decoder_dir):
        ckpt = torch.load(decoder_dir)
        self.ms_decoder.load_state_dict(ckpt['decoder'])
        self.ms_decoder.freeze_params() # freeze params to avoid backprop
