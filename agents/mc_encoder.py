import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models.mocap_solver import TS_AE, MC_AE
from models.mocap_solver.utils import Offset_loss, LBS_skel
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_encoder import MC_Dataset


class MC_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(MC_Agent, self).__init__(cfg, test, sweep)
        # torch.autograd.set_detect_anomaly(True)
        self.ts_checkpoint_dir = cfg.ts_model
        self.marker_weights = torch.tensor(cfg.marker_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.offset_weights = torch.tensor(np.load(cfg.offset_weights), dtype=torch.float32, device=self.device)
        self.offset_weights = self.offset_weights.view(*self.offset_weights.shape, 1)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.betas = cfg.loss.betas

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.model = MC_AE(self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims).to(self.device)

        self.ts_model = TS_AE(self.edges).to(self.device)
        self.load_ts_model()
        self.ts_model.eval()

    def load_data(self):
        self.train_dataset = MC_Dataset(data_dir=self.cfg.train_filenames)
        self.val_dataset = MC_Dataset(data_dir=self.cfg.val_filenames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, X_c, X_t):
        bs = X_c.shape[0]
        offsets = self.ts_model(X_t.view(bs, -1))
        Y_t = offsets[-1]
        _, Y_c = self.model(X_c.view(bs, -1), offsets)

        Y_c = Y_c.view(bs, self.num_markers, self.num_joints, 3)
        Y_t = Y_t.view(bs, self.num_joints, 3)
        X_t = X_t.view(bs, self.num_joints, 3)

        Y = LBS_skel(self.skinning_w, Y_c, Y_t)
        X = LBS_skel(self.skinning_w, X_c, X_t)
        return Y_c, X, Y

    def warmup(self, epochs=10):
        print("Starting Warmup")
        with torch.no_grad():
            for _ in range(epochs):
                for batch_idx, (X_c, X_t) in enumerate(self.train_data_loader):
                    X_c, X_t = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device)
                    self.run_batch(X_c, X_t)
        print("Warmup Done!")

    def train_per_epoch(self, epoch):
        if epoch == 0:
            self.warmup()
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_mp_err = 0
        n = 0

        # self.model.change_param_states(True)
        self.model.train()
        for batch_idx, (X_c, X_t) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X_c, X_t = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device)
            Y_c, X, Y = self.run_batch(X_c, X_t)
            loss = self.criterion(Y_c, X_c, Y, X)
            total_loss += (loss.item() * bs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mp_err = torch.norm(X.detach().clone() - Y.detach().clone(), 2, dim=-1) # n x m
            total_mp_err += mp_err.sum() / self.num_joints

            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, mp_err={2:.4f}mm".format(epoch, 1000*loss.item(), 1000*mp_err.mean())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, total_mp_err)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Val:   Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, mpe={2:.4f}mm".format(epoch, 1000*total_loss, 1000*total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_mp_err = 0
        n = 0

        # self.model.change_param_states(False)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X_c, X_t = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device)
                Y_c, X, Y = self.run_batch(X_c, X_t)
                loss = self.criterion(Y_c, X_c, Y, X)
                total_loss += (loss.item() * bs)

                mp_err = torch.norm(X.detach().clone() - Y.detach().clone(), 2, dim=-1) # n x m
                total_mp_err += mp_err.sum() / self.num_joints

                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, mp_err={2:.4f}mm".format(epoch, 1000*loss.item(), 1000*mp_err.mean())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, total_mp_err)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Val:   Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Val: Epoch={0:04d}, loss={1:.4f}, mpe={2:.4f}mm".format(epoch, 1000 * total_loss, 1000 * total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def build_loss_function(self):
        b3, b4 = self.betas
        return Offset_loss(self.marker_weights , self.offset_weights, b3, b4)

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

    def wandb_summary(self, training, total_loss, total_mp_err):
        if not training:
            wandb.log({'Validation Loss': total_loss})
            wandb.log({'Validation Marker Position Error (mm)': 1000*total_mp_err})
        else:
            wandb.log({'Training Loss': total_loss})
            wandb.log({'Training Marker Position Error (mm)': 1000*total_mp_err})
