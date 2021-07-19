import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import MoCap, collate_fn
from models.baseline import Baseline

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_dir = cfg.model_dir

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
        self.best_acc = 0

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
                loss, acc = self.val_per_epoch(epoch)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_acc = acc
                    self.save_model(epoch)

        self.train_writer.close()
        self.val_writer.close()

    def train_per_epoch(self, epoch):
        pass

    def val_per_epoch(self, epoch):
        pass

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr)
        elif optimizer == "sgd":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr)

    def build_loss_function(self):
        return nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    def save_model(self):
        ckpt = {'model': self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "best_acc": self.best_acc,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model']).to(self.device)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_loss = ckpt['best_loss']
        self.best_acc = ckpt['best_acc']

        return ckpt['epoch']
