import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import wandb

from agents.base_agent import BaseAgent
import models
from models.loss import Holden_loss
from datasets.robust_solver import RS_Dataset, RS_Test_Dataset

from tools.utils import LBS, corrupt, preweighted_Z, xform_to_mat44, symmetric_orthogonalization
from tools.utils import svd_rot as svd_solver
from tools.preprocess import weight_assign
from tools.statistics import *
from tools.transform import transformation_diff


class RS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(RS_Agent, self).__init__(cfg, test)
        self.is_sweep = sweep
        self.conv_to_m = 0.56444#0.57803

        self.user_weights_rot = cfg.user_weights_rotation
        self.user_weights_t = cfg.user_weights_translation
        self.w = weight_assign(cfg.joint_to_marker, cfg.num_markers, cfg.num_joints)

        self.w = self.w.to(self.device)
        self.user_weights_rot = torch.tensor(self.user_weights_rot, device=self.device)[None, ..., None, None] / 3
        self.user_weights_t = torch.tensor(self.user_weights_t, device=self.device)[None, ..., None, None]
        self.sampler = self.load_sampler()

    def load_sampler(self):
        mu = np.load(self.cfg.sampler.mean_fname).reshape((self.num_markers,self.num_joints,3))
        scale_tril = np.load(self.cfg.sampler.cholesky_fname)
        mu = torch.FloatTensor(mu).to(self.device).view(-1)
        scale_tril = torch.FloatTensor(scale_tril).to(self.device)

        return torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=scale_tril)

    def build_model(self):
        used = self.cfg.model.used.lower()
        if used == "baseline":
            hidden_size = self.cfg.model.baseline.hidden_size
            use_svd = self.cfg.model.baseline.use_svd
            num_layers = self.cfg.model.baseline.num_layers
            
            self.model = models.Baseline(self.num_markers, self.num_joints, hidden_size, num_layers, use_svd)
        elif used == "least_square":
            w = weight_assign('dataset/joint_to_marker_three2one.txt').to(self.device)
            self.model = models.LS_solver(self.num_joints, w, self.device)
        else:
            raise NotImplementedError

        self.model.to(self.device)

    def load_data(self):
        self.train_dataset = RS_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = RS_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.cfg.batch_size
        self.val_steps = len(self.val_dataset) // self.cfg.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        self.best_loss = float("inf")
        val_loss_f = []
        train_loss_f = []

        wandb.init(config=self.default_cfg(), project='denoising', entity='mocap')
        if self.is_sweep:
            sweep_config = wandb.config
            model_used = self.cfg.model.used.lower()
            optimizer_used = self.cfg.optimizer.used.lower()
            scheduler_used = self.cfg.lr_scheduler.used
            if model_used == "baseline":
                self.cfg.model.baseline.use_svd = sweep_config.use_svd
            if optimizer_used == "amsgrad":
                self.cfg.optimizer.AmsGrad.lr = sweep_config.lr
            if scheduler_used == "ExponentialLR":
                self.cfg.lr_scheduler.ExponentialLR.decay = sweep_config.decay

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        self.load_data()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        self.scheduler = self.lr_scheduler(last_epoch)

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):            
            _, msg = self.train_per_epoch(epoch)
            train_loss_f.append(msg)
            loss, msg = self.val_per_epoch(epoch)
            val_loss_f.append(msg)
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_model(epoch)

            self.scheduler.step()

        with open(self.cfg.logs_dir + "train_loss.txt", "w+") as f:
            for msg in train_loss_f:
                f.write(msg + "\n")
        with open(self.cfg.logs_dir + "val_loss.txt", "w+") as f:
            for msg in val_loss_f:
                f.write(msg + "\n")

        self.train_writer.close()
        self.val_writer.close()

    def run_batch(self, Y, Z, avg_bone, bs, sample_markers, corrupt_markers):
        if sample_markers:
            Z = self.sampler.sample((bs, )).view(-1, self.num_markers, self.num_joints, 3)

        X = LBS(self.w, Y, Z, device=self.device)
        if corrupt_markers:
            beta = 0.05 / (self.conv_to_m * avg_bone)
            X_hat = corrupt(X, beta=beta.view(-1), device=self.device)
        else:
            X_hat = X

        if self.cfg.model.used.lower() == "least_square":
            X = X_hat
        else:
            X = normalize_X(X_hat, X)
            Z_pw = preweighted_Z(self.w, Z)
            Z = normalize_Z_pw(Z_pw)

        Y_hat = self.model(X, Z).view(bs, self.num_joints, 3, 4)
        if self.cfg.model.used.lower() != "least_square":
            Y_hat = denormalize_Y(Y_hat, Y)
        return Y_hat

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_loss_rot = 0
        total_loss_tr = 0
        total_angle_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        total_translation_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        n = 0

        self.model.train()
        for batch_idx, (Y, Z, _, avg_bone) in enumerate(self.train_data_loader):
            bs = Y.shape[0]
            n += bs
            Y, Z, avg_bone = Y.to(torch.float32), Z.to(torch.float32), avg_bone.to(torch.float32)
            Y, Z, avg_bone = Y.to(self.device), Z.to(self.device), avg_bone.to(self.device).squeeze(-1)

            Y_hat = self.run_batch(Y, Z, avg_bone, bs,\
                                    sample_markers=self.cfg.training_settings.train_set.sample,\
                                    corrupt_markers=self.cfg.training_settings.train_set.corrupt)

            loss_rot, loss_tr = self.criterion(Y, Y_hat)
            loss = loss_tr + loss_rot

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_rot += loss_rot.item()
            total_loss_tr += loss_tr.item()

            avg_bone = avg_bone[..., None, None]
            Y_h = Y_hat.detach().clone()
            Y_ = Y.detach().clone()
            Y_h[..., 3] *= (self.conv_to_m * avg_bone * 100)
            Y_[..., 3] *= (self.conv_to_m * avg_bone * 100)
            
            Y_h_orth = Y_h.clone().view(-1, 3, 4)
            Y_h_orth[..., :3] = symmetric_orthogonalization(Y_h_orth[..., :3].clone()).clone()
            Y_h_orth = Y_h_orth.view(-1, self.num_joints, 3, 4)

            angle_diff, translation_diff = transformation_diff(Y_h_orth, Y_)
            total_angle_diff += torch.sum(torch.abs(angle_diff), axis=0)
            total_translation_diff += torch.sum(translation_diff, axis=0)

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, rot_loss={2:.4f}, t_loss={3:.4f}".format(epoch, loss.item() / bs, loss_rot.item() / bs, loss_tr.item() / bs)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_loss_rot /= n
        total_loss_tr /= n
        total_translation_diff /= n
        total_angle_diff *= 180 / (n * np.pi)
        losses = (total_loss, total_loss_rot, total_loss_tr)
        self.write_summary(self.train_writer, losses, total_angle_diff, total_translation_diff, epoch)
        self.wandb_summary(True, losses, total_angle_diff, total_translation_diff, epoch)

        tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}, rot_loss={2:.4f}, t_loss={3:.4f}".format(epoch, total_loss, total_loss_rot, total_loss_tr)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True)
        total_loss = 0
        total_loss_rot = 0
        total_loss_tr = 0
        total_angle_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        total_translation_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (Y, Z, _, avg_bone) in enumerate(self.val_data_loader):
                bs = Y.shape[0]
                n += bs
                Y, Z, avg_bone = Y.to(torch.float32).to(self.device), Z.to(torch.float32).to(self.device), avg_bone.to(torch.float32).to(self.device).squeeze(-1)

                Y_hat = self.run_batch(Y, Z, avg_bone, bs,\
                                    sample_markers=self.cfg.training_settings.val_set.sample,\
                                    corrupt_markers=self.cfg.training_settings.val_set.corrupt)

                loss_rot, loss_tr = self.criterion(Y, Y_hat)
                loss = loss_tr + loss_rot

                total_loss += loss.item()
                total_loss_rot += loss_rot.item()
                total_loss_tr += loss_tr.item()

                avg_bone = avg_bone[..., None, None]
                Y_h = Y_hat.detach().clone()
                Y_ = Y.detach().clone()
                Y_h[..., 3] *= (self.conv_to_m * avg_bone * 100)
                Y_[..., 3] *= (self.conv_to_m * avg_bone * 100)

                Y_h_orth = Y_h.clone().view(-1, 3, 4)
                Y_h_orth[..., :3] = symmetric_orthogonalization(Y_h_orth[..., :3].clone()).clone()
                Y_h_orth = Y_h_orth.view(-1, self.num_joints, 3, 4)

                angle_diff, translation_diff = transformation_diff(Y_h_orth, Y_)
                total_angle_diff += torch.sum(torch.abs(angle_diff), axis=0)
                total_translation_diff += torch.sum(translation_diff, axis=0)

                tqdm_update = "Epoch={0:04d},loss={1:.4f}, rot_loss={2:.4f}, t_loss={3:.4f}".format(epoch, loss.item() / bs, loss_rot.item() / bs, loss_tr.item() / bs)
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_loss_rot /= n
        total_loss_tr /= n
        total_translation_diff /= n
        total_angle_diff *= 180 / (n * np.pi)
        losses = (total_loss, total_loss_rot, total_loss_tr)
        self.write_summary(self.val_writer, losses, total_angle_diff, total_translation_diff, epoch)
        self.wandb_summary(False, losses, total_angle_diff, total_translation_diff, epoch)

        tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}, rot_loss={2:.4f}, t_loss={3:.4f}".format(epoch, total_loss, total_loss_rot, total_loss_tr)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def test_one_animation(self):
        self.test_dataset = RS_Test_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.test_filenames,\
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.test_steps = len(self.test_dataset)

        self.test_data_loader = DataLoader(self.test_dataset, batch_size=1, \
                                            shuffle=False, num_workers=8)

        self.build_model()
        self.model.to(self.device)
        self.criterion = self.build_loss_function()

        if self.cfg.model.used.lower() != "least_square":
            if os.path.exists(self.checkpoint_dir):
                self.load_model()
            else:
                print("There is no saved model")
                exit()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (Y, Z, F, avg_bone) in enumerate(self.test_data_loader):
                Y, Z, F, avg_bone = Y.to(torch.float32).to(self.device).squeeze(0), Z.to(torch.float32).to(self.device).squeeze(0),\
                                    F.to(torch.float32).to(self.device).squeeze(0).unsqueeze(1), avg_bone.to(torch.float32).to(self.device).squeeze(-1)
                bs = Y.shape[0]
                
                Y_hat = self.run_batch(Y, Z, avg_bone, bs,
                                    sample_markers=self.cfg.training_settings.val_set.sample,
                                    corrupt_markers=self.cfg.training_settings.val_set.corrupt)

                loss_rot, loss_tr = self.criterion(Y, Y_hat)
                loss_rot /= bs
                loss_tr /= bs
                loss = loss_tr + loss_rot

                Y_hat_4x4 = xform_to_mat44(Y_hat, self.device)
                Y_ = F @ Y_hat_4x4
                Y_ = Y_.cpu().detach().numpy()
                np.save(f"asd.npy", Y_)
                print("loss={0:.4f}".format(loss.item()))

    def lr_scheduler(self, last_epoch):
        scheduler = self.cfg.lr_scheduler.used
        if scheduler == "ExponentialLR":
            return ExponentialLR(optimizer=self.optimizer, gamma=self.cfg.lr_scheduler.ExponentialLR.decay)
        elif scheduler == "MultiStepLR":
            milestones = list(range(0, self.cfg.epochs, self.cfg.lr_scheduler.MultiStepLR.range))
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch-1)

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr)
        elif optimizer == "amsgrad":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.AmsGrad.lr,
                                    weight_decay=self.cfg.optimizer.AmsGrad.weight_decay, amsgrad=True)

    def build_loss_function(self):
        return Holden_loss(self.user_weights_t, self.user_weights_rot)

    def default_cfg(self):
        return {
            "use_svd": self.cfg.model.baseline.use_svd,
            "lr": self.cfg.optimizer.AmsGrad.lr,
            "decay": self.cfg.lr_scheduler.ExponentialLR.decay
        }
