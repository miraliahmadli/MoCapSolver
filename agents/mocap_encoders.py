import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import AE, AE_loss
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import AE_Dataset
from tools.transform import quaternion_to_matrix, matrix_to_axis_angle, matrix_to_quaternion


class EncoderAgent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(EncoderAgent, self).__init__(cfg, test, sweep)
        torch.autograd.set_detect_anomaly(True)
        self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.marker_weights = torch.tensor(cfg.marker_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.offset_weights = torch.tensor(np.load(cfg.offset_weights), dtype=torch.float32, device=self.device)
        self.offset_weights = self.offset_weights.view(*self.offset_weights.shape, 1)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.window_size = cfg.window_size
        self.betas = cfg.loss.betas

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.model = AE(self.edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
                        offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7]).to(self.device)

    def load_data(self):
        self.train_dataset = AE_Dataset(data_dir=self.cfg.train_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames)
        self.val_dataset = AE_Dataset(data_dir=self.cfg.val_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, X_c, X_t, X_m):
        bs = X_c.shape[0]
        Y_c, Y_t, Y_m = self.model(X_c.view(bs, -1), X_t.view(bs, -1), X_m.view(bs, -1, self.window_size))
        Y_c = Y_c.view(bs, self.num_markers, self.num_joints, 3)
        Y_t = Y_t.view(bs, self.num_joints, 3)
        return Y_c, Y_t, Y_m

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        n = 0

        self.model.train()
        for batch_idx, (X_c, X_t, X_m) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device), X_m.to(torch.float32).to(self.device)
            
            Y_c, Y_t, Y_m = self.run_batch(X_c, X_t, X_m)

            loss, loss_c, loss_t, loss_m = self.criterion((X_c, X_t, X_m), (Y_c, Y_t, Y_m))
            total_loss_c += (loss_c.item() * bs)
            total_loss_t += (loss_t.item() * bs)
            total_loss_m += (loss_m.item() * bs)
            total_loss += (loss.item() * bs)

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

            # tqdm_update = f"Epoch={epoch:04d}, offset_loss={loss_c.item() / bs:.4f}, skeletal_loss={loss_t.item() / bs:.4f}, motion_loss={loss_m.item() / bs:.4f}"
            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss_c /= n
        total_loss_t /= n
        total_loss_m /= n
        total_loss /= n

        total_angle_err /= n
        total_jp_err /= n
        # self.write_summary(self.train_writer, total_loss, epoch)
        # self.wandb_summary(True, total_loss, epoch)

        # tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}".format(epoch, total_loss, total_angle_err, total_jp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device), X_m.to(torch.float32).to(self.device)

                Y_c, Y_t, Y_m = self.run_batch(X_c, X_t, X_m)

                loss, loss_c, loss_t, loss_m = self.criterion((X_c, X_t, X_m), (Y_c, Y_t, Y_m))
                total_loss_c += (loss_c.item() * bs)
                total_loss_t += (loss_t.item() * bs)
                total_loss_m += (loss_m.item() * bs)
                total_loss += (loss.item() * bs)

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

                # tqdm_update = f"Epoch={epoch:04d}, offset_loss={loss_c.item() / bs:.4f}, skeletal_loss={loss_t.item() / bs:.4f}, motion_loss={loss_m.item() / bs:.4f}"
                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss_c /= n
        total_loss_t /= n
        total_loss_m /= n
        total_loss /= n

        total_angle_err /= n
        total_jp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Val:   Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Val: Epoch={0:04d}, loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}".format(epoch, total_loss, total_angle_err, total_jp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def build_loss_function(self):
        return AE_loss(self.joint_topology, self.marker_weights , self.joint_weights, self.offset_weights, self.betas, self.skinning_w)

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
