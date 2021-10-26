import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models.mocap_solver import TS_AE
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_encoder import TS_Dataset


class TS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(TS_Agent, self).__init__(cfg, test, sweep)
        # torch.autograd.set_detect_anomaly(True)
        self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device).view(-1, 1)

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.model = TS_AE(self.edges).to(self.device)

    def load_data(self):
        self.train_dataset = TS_Dataset(data_dir=self.cfg.train_filenames)
        self.val_dataset = TS_Dataset(data_dir=self.cfg.val_filenames)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, X_t):
        bs = X_t.shape[0]
        _, _, Y_t = self.model(X_t.view(bs, -1))
        Y_t = Y_t.view(bs, self.num_joints, 3)
        return Y_t

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_jp_err = 0
        n = 0

        self.model.train()
        for batch_idx, X_t in enumerate(self.train_data_loader):
            bs = X_t.shape[0]
            n += bs
            X_t = X_t.to(torch.float32).to(self.device)
            Y_t = self.run_batch(X_t)

            loss  = self.criterion(Y_t, X_t)
            total_loss += (loss.item() * bs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
            total_jp_err += jp_err.sum() / self.num_joints

            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, jp_err={2:.4f}mm".format(epoch, 1000*loss.item(), 1000*jp_err.mean())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_jp_err /= n
        # self.write_summary(self.train_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, total_jp_err)

        # tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, jpe={2:.4f}mm".format(epoch, 1000*total_loss, 1000*total_jp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_jp_err = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, X_t in enumerate(self.val_data_loader):
                bs = X_t.shape[0]
                n += bs
                X_t = X_t.to(torch.float32).to(self.device)
                Y_t = self.run_batch(X_t)

                loss  = self.criterion(Y_t, X_t)
                total_loss += (loss.item() * bs)

                jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
                total_jp_err += jp_err.sum() / self.num_joints

                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, jp_err={2:.4f}mm".format(epoch, 1000*loss.item(), 1000*jp_err.mean())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_jp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, total_jp_err)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_update = "Val:   Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}".format(epoch, total_loss, total_loss_c, total_loss_t, total_loss_m)
        tqdm_update = "Val: Epoch={0:04d}, loss={1:.4f}, jpe={2:.4f}mm".format(epoch, 1000*total_loss, 1000*total_jp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def build_loss_function(self):
        return nn.SmoothL1Loss()

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

        print(f"Pretrained model loaded: Epoch{ckpt['epoch']}")
        return ckpt['epoch']

    def wandb_summary(self, training, total_loss, total_jp_err):
        if not training:
            wandb.log({'Validation Loss': total_loss})
            wandb.log({'Validation Joint Position Error (mm)': 1000*total_jp_err})
        else:
            wandb.log({'Training Loss': total_loss})
            wandb.log({'Training Joint Position Error (mm)': 1000*total_jp_err})
