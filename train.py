import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import MoCap, collate_fn
from models.baseline import Baseline
from tools.utils import svd_rot_torch as svd_solver


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_dir = cfg.model_dir
        self.gamma = cfg.loss.gamma
        self.user_weights = cfg.user_weights

        self.device = torch.device('cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        print(self.device)
        if self.cfg.use_gpu and torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        self.model = None

    def build_model(self):
        num_markers = self.cfg.num_markers
        num_joints = self.cfg.num_joints

        used = self.cfg.model.used.lower()
        if used == "baseline":
            hidden_size = self.cfg.model.baseline.hidden_size
            use_svd = self.cfg.model.baseline.use_svd
            num_layers = self.cfg.model.baseline.num_layers
            
            self.model = Baseline(num_markers, num_joints, hidden_size, num_layers, use_svd)
        elif used == "least_square":
            # TODO: train loop for LS problem
            self.model = None
        else:
            raise NotImplementedError

    def load_data(self):
        self.train_dataset = MoCap(data_dir=self.cfg.train_datadir , fnames=self.cfg.train_filenames)
        self.val_dataset = MoCap(data_dir=self.cfg.val_datadir , fnames=self.cfg.var_filenames)

        self.train_steps = len(self.train_dataset) // self.cfg.batch_size
        self.val_steps = len(self.val_dataset) // self.cfg.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,\
                                            collate_fn=collate_fn, shuffle=True, num_workers=2)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,\
                                            collate_fn=collate_fn, shuffle=False, num_workers=2)

    def train(self):
        self.best_loss = float("inf")

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        self.load_data()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):
            self.train_per_epoch(epoch)
            if epoch > 1:
                loss = self.val_per_epoch(epoch)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model(epoch)

        self.train_writer.close()
        self.val_writer.close()

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True)
        total_loss = 0

        self.model.train()
        for batch_idx, (X, Y, Z, F) in enumerate(self.train_data_loader):
            X, Y, Z, F = X.to(torch.float32), Y.to(torch.float32),\
                        Z.to(torch.float32), F.to(torch.float32)
            X, Y, Z, F = X.to(self.device), Y.to(self.device), Z.to(self.device), F.to(self.device)

            # TODO:
            # do preprocessing and get corrupted X and preweighted Z
            self.optimizer.zero_grad()
            Y_hat = self.model(X, Z)

            loss = self.user_weights * self.criterion(Y_hat, Y) + self.l2_regularization()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, l1_loss={2:.4f}, reg={3:.4f}".format(epoch, loss.item(), l1_loss, l2_reg)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        self.train_writer.add_scalar('Loss', total_loss, epoch)
        print("epoch", epoch, "loss:", total_loss)

        tqdm_batch.close()
        return total_loss

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True)
        total_loss = 0

        self.model.eval()
        for batch_idx, (X, Y, Z, F) in enumerate(self.val_data_loader):
            X, Y, Z, F = X.to(torch.float32), Y.to(torch.float32),\
                        Z.to(torch.float32), F.to(torch.float32)
            X, Y, Z, F = X.to(self.device), Y.to(self.device), Z.to(self.device), F.to(self.device)

            # TODO:
            # do preprocessing and get corrupted X and preweighted Z
            self.optimizer.zero_grad()
            Y_hat = self.model(X, Z)

            l1_loss = self.user_weights * self.criterion(Y_hat, Y)
            l2_reg = self.l2_regularization()
            loss = l1_loss + l2_reg
            total_loss += loss.item()

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, l1_loss={2:.4f}, reg={3:.4f}".format(epoch, loss.item(), l1_loss, l2_reg)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        self.val_writer.add_scalar('Loss', total_loss, epoch)
        print("epoch", epoch, "loss:", total_loss)

        tqdm_batch.close()
        return total_loss

    def l2_regularization(self):
        l2 = torch.tensor(0)

        for param in self.model.parameters():
            l2 += torch.norm(param, 2)**2

        return l2 * self.gamma

    def ls_solver(self):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True)
        self.build_model()
        self.criterion = self.build_loss_function()
        self.load_data()

        total_loss = 0
        for batch_idx, (X, Y, Z, F) in enumerate(self.val_data_loader):
            X, Y, Z, F = X.to(torch.float32), Y.to(torch.float32),\
                        Z.to(torch.float32), F.to(torch.float32)
            X, Y, Z, F = X.to(self.device), Y.to(self.device), Z.to(self.device), F.to(self.device)

            # TODO:
            # do preprocessing and get corrupted X and preweighted Z
            Y_hat = self.model(X, Z)

            l1_loss = self.user_weights * self.criterion(Y_hat, Y)
            l2_reg = self.l2_regularization()
            loss = l1_loss + l2_reg
            total_loss += loss.item()

            tqdm_update = "Loss={0:.4f}, L1={1:.4f}, reg={2:.4f}, Total={3:.4f}".format(loss.item(), l1_loss, l2_reg, total_loss)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        print("loss:", total_loss)
        tqdm_batch.close()

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr)
        elif optimizer == "amsgrad":
            return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.AmsGrad.lr,
                                    weight_decay=self.cfg.optimizer.AmsGrad.weight_decay, amsgrad=True)

    def build_loss_function(self):
        return nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    def save_model(self):
        ckpt = {'model': self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model']).to(self.device)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_loss = ckpt['best_loss']

        return ckpt['epoch']
