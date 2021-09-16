import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import AE, MocapSolver, MS_loss
from models.mocap_solver import Decoder, MSD, FK, LBS_motion
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import MS_Dataset
from tools.transform import quaternion_to_matrix, matrix_to_axis_angle, matrix_to_quaternion
from tools.utils import xform_to_mat44


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(MS_Agent, self).__init__(cfg, test, sweep)
        torch.autograd.set_detect_anomaly(True)
        self.train_decoder = cfg.model.train_decoder
        self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.marker_weights = torch.tensor(cfg.marker_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.offset_weights = torch.tensor(np.load(cfg.offset_weights), dtype=torch.float32, device=self.device)
        self.offset_weights = self.offset_weights.view(*self.offset_weights.shape, 1)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.alphas = cfg.loss.alphas
        self.window_size = cfg.window_size

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        # self.auto_encoder = AE(self.edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
        #                        offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])#.to(self.device)
        # self.mocap_solver = MocapSolver(self.num_markers, self.window_size, 1024,
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
        self.train_dataset = MS_Dataset(data_dir=self.cfg.train_filenames, window_size=self.window_size, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)
        self.val_dataset = MS_Dataset(data_dir=self.cfg.val_filenames, window_size=self.window_size, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, X, F):
        Y_c, Y_t, Y_m = self.model(X)

        # skinning
        j_rot, j_tr = FK(self.joint_topology, Y_m, Y_t)
        j_xform = torch.cat([j_rot, j_tr.unsqueeze(-1)], dim=-1)
        Y = LBS_motion(self.skinning_w, Y_c, j_xform)

        # j_rot_gt, j_tr_gt = FK(self.joint_topology, X_m, X_t)
        # j_xform_gt = torch.cat([j_rot_gt, j_tr_gt.unsqueeze(-1)], dim=-1)

        # denorm
        # nans = ~((Y != 0.0).any(axis=-1))
        # Y_denorm = F[..., :3].unsqueeze(2) @ Y[..., None] + F[..., 3, None].unsqueeze(2)
        # Y_denorm[nans] = torch.zeros((3,1)).to(torch.float32).to(self.device)
        # Y_denorm = Y_denorm.squeeze(-1)

        # from tools.viz import visualize
        # visualize(Xs=10*Y[0].detach().cpu().numpy()[..., [0, 2, 1]])
        # Ys = 10 * torch.cat((j_xform[:1], j_xform_gt[:1]), 0).detach().cpu().numpy()
        # visualize(Ys=Ys)
        # Xs = 10 * torch.cat((Y[:1], X[:1]), 0).detach().cpu().numpy()
        # visualize(Xs=Xs, fps_vid=120, colors_X=[[0, 255, 0], [0, 0, 255]])
        # exit()

        return Y_c, Y_t, Y_m, Y

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        total_mp_err = 0
        n = 0

        self.model.train()
        # if self.train_decoder:
        #     self.ms_decoder.train()
        for batch_idx, (X_c, X_t, X_m, X, X_clean, F) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X, F, X_clean = X.to(torch.float32).to(self.device), F.to(torch.float32).to(self.device), X_clean.to(torch.float32).to(self.device)
            X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

            Y_c, Y_t, Y_m, Y = self.run_batch(X, F)

            losses = self.criterion((Y_c, Y_t, Y_m, Y), (X_c, X_t, X_m, X_clean))
            loss, loss_c, loss_t, loss_m, loss_marker = losses
            total_loss += loss.item() * bs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            quat_X = X_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
            quat_Y = Y_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
            R_x = quaternion_to_matrix(quat_X)
            R_y = quaternion_to_matrix(quat_Y)
            R = R_y.transpose(-2, -1) @ R_x
            axis_angle = matrix_to_axis_angle(R)
            angle_diff = axis_angle[..., -1] / np.pi * 180 # n x t x j
            total_angle_err += torch.sum(torch.abs(angle_diff)) / (self.window_size * self.num_joints)

            jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
            total_jp_err += jp_err.sum() / self.num_joints

            mp_err = torch.norm(X_clean.detach().clone().view(-1, self.num_markers, 3) - Y.detach().clone().view(-1, self.num_markers, 3), 2, dim=-1) # (n x t) x m
            total_mp_err += mp_err.sum() / (self.num_markers * self.window_size)

            # tqdm_update = "Epoch={0:04d}, loss={1:.4f}, angle_diff={2:.4f}, jpe={3:.4f}, mpe={4:4f}".format(epoch, 1000*loss.item() / bs, torch.abs(angle_diff).mean(), jp_err.mean(), mp_err.mean())
            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}, loss_marker={5:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item(), loss_marker.item())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jp_err /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}, mpe={4:4f}".format(epoch, total_loss, total_angle_err, total_jp_err, total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        total_mp_err = 0
        n = 0

        self.model.eval()
        # if self.train_decoder:
        #     self.ms_decoder.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m, X, X_clean, F) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X, F, X_clean = X.to(torch.float32).to(self.device), F.to(torch.float32).to(self.device), X_clean.to(torch.float32).to(self.device)
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

                Y_c, Y_t, Y_m, Y = self.run_batch(X, F)
                losses = self.criterion((Y_c, Y_t, Y_m, Y), (X_c, X_t, X_m, X_clean))
                loss, loss_c, loss_t, loss_m, loss_marker = losses
                total_loss += loss.item() * bs

                quat_X = X_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
                quat_Y = Y_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
                R_x = quaternion_to_matrix(quat_X)
                R_y = quaternion_to_matrix(quat_Y)
                R = R_y.transpose(-2, -1) @ R_x
                axis_angle = matrix_to_axis_angle(R)
                angle_diff = axis_angle[..., -1] / np.pi * 180 # (n x t) x j
                total_angle_err += torch.sum(torch.abs(angle_diff)) / (self.window_size * self.num_joints)

                jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
                total_jp_err += jp_err.sum() / self.num_joints

                mp_err = torch.norm(X_clean.detach().clone().view(-1, self.num_markers, 3) - Y.detach().clone().view(-1, self.num_markers, 3), 2, dim=-1) # (n x t) x m
                total_mp_err += mp_err.sum() / (self.num_markers * self.window_size)

                # tqdm_update = "Epoch={0:04d}, loss={1:.4f}, angle_diff={2:.4f}, mpe={4:4f}".format(epoch, loss.item(), torch.abs(angle_diff).mean(), jp_err.mean(), mp_err.mean())
                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}, loss_marker={5:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item(), loss_marker.item())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jp_err /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}, mpe={4:4f}".format(epoch, total_loss, total_angle_err, total_jp_err, total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def test_one_animation(self):
        pass

    def build_loss_function(self):
        return MS_loss(self.joint_weights, self.marker_weights, self.offset_weights, self.alphas)

    def load_decoder(self, decoder_dir):
        ckpt = torch.load(decoder_dir)
        self.ms_decoder.load_state_dict(ckpt['decoder'])
        self.ms_decoder.freeze_params() # freeze params to avoid backprop
