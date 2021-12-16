import math
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models.mocap_solver import TS_AE, Motion_AE
from models.mocap_solver.utils import Motion_loss, FK
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_encoder import Motion_Dataset
from tools.transform import quaternion_to_matrix, matrix_to_axis_angle


class Motion_Agent(BaseAgent):
    def __init__(self, cfg, test=False):
        super(Motion_Agent, self).__init__(cfg, test)
        self.ts_checkpoint_dir = cfg.ts_model

        self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.offset_weights = torch.tensor(np.load(cfg.offset_weights), dtype=torch.float32, device=self.device)
        self.offset_weights = self.offset_weights.view(*self.offset_weights.shape, 1)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.window_size = cfg.window_size
        self.betas = cfg.loss.betas

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.model = Motion_AE(self.edges, offset_channels=[1, 8], 
                                offset_joint_num=[self.num_joints, 7]).to(self.device)

        self.ts_model = TS_AE(self.edges).to(self.device)
        self.load_ts_model()
        self.ts_model.eval()

    def load_data(self):
        self.train_dataset = Motion_Dataset(data_dir=self.cfg.train_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames)
        self.val_dataset = Motion_Dataset(data_dir=self.cfg.val_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    def run_batch(self, X_t, X_m):
        bs = X_t.shape[0]
        offsets = self.ts_model(X_t.view(bs, -1))
        Y_t = offsets[-1]
        _, Y_m = self.model(X_m[:, :, 4:].transpose(-1, -2), [None]*3)

        Y_m = Y_m.transpose(-1, -2)
        Y_t = Y_t.view(bs, self.num_joints, 3)
        return Y_t, Y_m

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True)
        total_loss = 0
        total_angle_err = 0
        total_jpe = 0
        n = 0

        self.model.train()
        for batch_idx, (X_t, X_m) in enumerate(self.train_data_loader):
            bs = X_t.shape[0]
            n += bs
            X_t, X_m = X_t.to(torch.float32).to(self.device), X_m.to(torch.float32).to(self.device)

            first_rot = X_m[:, :, :4].detach().clone()
            Y_t, Y_m = self.run_batch(X_t, X_m)
            Y_m = torch.cat([first_rot, Y_m], -1)

            loss = self.criterion(Y_m, X_m, Y_t, X_t)
            total_loss += (loss.item() * bs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            R_y, global_tr_y = FK(self.joint_topology, Y_m.detach().clone(), Y_t.detach().clone())
            R_x, global_tr_x = FK(self.joint_topology, X_m.detach().clone(), X_t.detach().clone())
            
            jpe = torch.sqrt(torch.sum(torch.pow(global_tr_y - global_tr_x, 2), dim=-1))
            total_jpe += (torch.sum(jpe)   / (self.window_size * self.num_joints))

            # quat_X = X_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
            # quat_Y = Y_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
            # R_x = quaternion_to_matrix(quat_X)
            # R_y = quaternion_to_matrix(quat_Y)
            R = R_y.transpose(-2, -1) @ R_x
            axis_angle = matrix_to_axis_angle(R)
            angle_diff = torch.norm(axis_angle, p=2, dim=-1)
            angle_diff[angle_diff > math.pi] -= 2*math.pi
            angle_diff = torch.abs(angle_diff) / np.pi * 180
            total_angle_err += torch.sum(angle_diff) / (self.window_size * self.num_joints)

            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, joe={2:.4f}, jpe={3:.4f}".format(epoch, loss.item(), angle_diff.mean(), jpe.mean())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jpe /= n
        # self.write_summary(self.train_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, total_angle_err, total_jpe)

        # tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, angle_err={2:.4f}, jpe={3:.4f}".format(epoch, total_loss, total_angle_err, total_jpe)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_angle_err, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_err = 0
        total_jpe = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_t, X_m) in enumerate(self.val_data_loader):
                bs = X_t.shape[0]
                n += bs
                X_t, X_m = X_t.to(torch.float32).to(self.device), X_m.to(torch.float32).to(self.device)
                first_rot = X_m[:, :, :4].detach().clone()
                Y_t, Y_m = self.run_batch(X_t, X_m)
                Y_m = torch.cat([first_rot, Y_m], -1)

                loss = self.criterion(Y_m, X_m, Y_t, X_t)
                total_loss += (loss.item() * bs)

                R_y, global_tr_y = FK(self.joint_topology, Y_m.detach().clone(), Y_t.detach().clone())
                R_x, global_tr_x = FK(self.joint_topology, X_m.detach().clone(), X_t.detach().clone())
                
                jpe = torch.sqrt(torch.sum(torch.pow(global_tr_y - global_tr_x, 2), dim=-1))
                total_jpe += (torch.sum(jpe)  / (self.window_size * self.num_joints))

                # quat_X = X_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
                # quat_Y = Y_m[...,:-3].detach().clone().view(bs, self.window_size, self.num_joints, 4)
                # R_x = quaternion_to_matrix(quat_X)
                # R_y = quaternion_to_matrix(quat_Y)
                R = R_y.transpose(-2, -1) @ R_x
                axis_angle = matrix_to_axis_angle(R)
                angle_diff = torch.norm(axis_angle, p=2, dim=-1)
                angle_diff[angle_diff > math.pi] -= 2*math.pi
                angle_diff = torch.abs(angle_diff) / np.pi * 180
                total_angle_err += torch.sum(angle_diff) / (self.window_size * self.num_joints)

                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, joe={2:.4f}, jpe={3:.4f}".format(epoch, loss.item(), angle_diff.mean(), jpe.mean())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jpe /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, total_angle_err, total_jpe)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Val:   Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Val: Epoch={0:04d}, loss={1:.4f}, angle_err={2:4f}, jpe={3:.4f}".format(epoch, total_loss, total_angle_err, total_jpe)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_angle_err, message

    def build_loss_function(self):
        b1, b2 = self.betas
        return Motion_loss(self.joint_topology, self.joint_weights, b1, b2)

    def save_model(self, epoch):
        ckpt = {'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.encoder.load_state_dict(ckpt["encoder"])
        self.model.decoder.load_state_dict(ckpt['decoder'])

        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']

    def load_ts_model(self):
        ckpt = torch.load(self.ts_checkpoint_dir)
        self.ts_model.encoder.load_state_dict(ckpt["encoder"])
        self.ts_model.decoder.load_state_dict(ckpt['decoder'])
        print(f"Loading TS Model: Epoch #{ckpt['epoch']}")
        return

    def wandb_summary(self, training, total_loss, total_angle_err, total_jpe):
        if not training:
            wandb.log({'Validation Loss': total_loss})
            wandb.log({'Validation Angle Error (deg)': total_angle_err})
            wandb.log({'Validation Joint Position Error (mm)': 1000*total_jpe})
        else:
            wandb.log({'Training Loss': total_loss})
            wandb.log({'Training Angle Error (deg)': total_angle_err})
            wandb.log({'Training Joint Position Error (mm)': 1000*total_jpe})
