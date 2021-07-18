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

    def build_model(self):
        pass

    def load_data(self):
        pass

    def train(self):
        pass

    def train_per_epoch(self):
        pass

    def val_per_epoch(self):
        pass

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr)
        elif optimizer == "sgd":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr)

    def build_loss_function(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
