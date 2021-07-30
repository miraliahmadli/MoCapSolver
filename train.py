import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR 

from mocap_dataset import MoCap
from models.baseline import Baseline
from tools.utils import svd_rot_torch as svd_solver
from tools.utils import LBS_torch as LBS
from tools.utils import corrupt_torch as corrupt
from tools.utils import preweighted_Z, xform_to_mat44_torch
from tools.preprocess import weight_assign
from tools.statistics import *
from tools.transform import transformation_diff


class Agent:
    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.checkpoint_dir = cfg.model_dir
        self.is_test = test
        self.conv_to_m = 0.056444

        self.num_markers = cfg.num_markers
        self.num_joints = cfg.num_joints
        self.batch_size = cfg.batch_size

        self.user_weights = cfg.user_weights
        self.w = weight_assign(cfg.joint_to_marker, cfg.num_markers, cfg.num_joints)

        self.device = torch.device('cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        print(self.device)
        if self.cfg.use_gpu and torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        self.w = self.w.to(self.device)
        self.model = None

    def build_model(self):
        used = self.cfg.model.used.lower()
        if used == "baseline":
            hidden_size = self.cfg.model.baseline.hidden_size
            use_svd = self.cfg.model.baseline.use_svd
            num_layers = self.cfg.model.baseline.num_layers
            
            self.model = Baseline(self.num_markers, self.num_joints, hidden_size, num_layers, use_svd)
        elif used == "least_square":
            # TODO: train loop for LS problem
            self.model = None
        else:
            raise NotImplementedError
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def load_data(self):
        self.train_dataset = MoCap(csv_file=self.cfg.csv_file , fnames=self.cfg.train_filenames,\
                                num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = MoCap(csv_file=self.cfg.csv_file , fnames=self.cfg.val_filenames,\
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.cfg.batch_size
        self.val_steps = len(self.val_dataset) // self.cfg.batch_size

        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         self.train_dataset, num_replicas=torch.cuda.device_count() )
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         self.val_dataset, shuffle=False, num_replicas=torch.cuda.device_count())

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)#, sampler=train_sampler)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)#, sampler=val_sampler)

    def train(self):
        self.best_loss = float("inf")
        val_loss_f = []
        train_loss_f =[]

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        self.load_data()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg.optimizer.Step_LR,\
                                    gamma=0.1, last_epoch=last_epoch-1)
        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):
            self.scheduler.step()
            _, msg = self.train_per_epoch(epoch)
            train_loss_f.append(msg)
            if epoch > 1:
                loss, msg = self.val_per_epoch(epoch)
                val_loss_f.append(msg)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model(epoch)

        with open(self.cfg.logs_dir + "train_loss.txt", "w+") as f:
            for msg in train_loss_f:
                f.write(msg + "\n")
        with open(self.cfg.logs_dir + "val_loss.txt", "w+") as f:
            for msg in val_loss_f:
                f.write(msg + "\n")

        self.train_writer.close()
        self.val_writer.close()

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        total_translation_loss = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        n = 0

        self.model.train()
        for batch_idx, (Y, Z, _, avg_bone) in enumerate(self.train_data_loader):
            bs = Y.shape[0]
            n += bs
            Y, Z, avg_bone = Y.to(torch.float32), Z.to(torch.float32), avg_bone.to(torch.float32)
            Y, Z, avg_bone = Y.to(self.device), Z.to(self.device), avg_bone.to(self.device).squeeze(-1)

            z_mu, z_cov = get_stat_Z(Z)
            Z_sample = sample_Z(z_mu, z_cov, bs).view(-1, self.num_markers, self.num_joints, 3)

            X = LBS(self.w, Y, Z_sample)

            beta = 0.5 / (self.conv_to_m * avg_bone)
            X_hat = corrupt(X, beta=beta)
            Z_pw = preweighted_Z(self.w, Z_sample)

            X = normalize_X(X_hat, X)
            Z = normalize_Z_pw(Z_pw)

            Y_hat = self.model(X, Z).view(bs, self.num_joints, 3, 4)
            Y_hat = denormalize_Y(Y_hat, Y)

            loss = self.user_weights * self.criterion(Y_hat, Y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() #/ (self.num_joints * 3 * 4)
            angle_diff, translation_diff = transformation_diff(Y_hat, Y)
            total_angle_diff += torch.sum(torch.abs(angle_diff), axis=0)
            total_translation_loss += torch.sum(translation_diff, axis=0)

            tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_translation_loss /= n
        total_angle_diff /= n
        self.train_writer.add_scalar('Loss', total_loss, epoch)
        for i in range(self.num_joints):
            self.train_writer.add_scalar(f'joint_{i+1}: avg angle diff', (total_angle_diff[i] / np.pi) * 180, epoch)
            self.train_writer.add_scalar(f'joint_{i+1}: avg translation diff', total_translation_loss[i], epoch)

        tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        message = f"epoch: {epoch}, loss: {total_loss}"
        tqdm_batch.close()
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True)
        total_loss = 0
        total_angle_diff = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        total_translation_loss = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (Y, Z, _, avg_bone) in enumerate(self.val_data_loader):
                bs = Y.shape[0]
                n += bs
                Y, Z, avg_bone = Y.to(torch.float32).to(self.device), Z.to(torch.float32).to(self.device), avg_bone.to(torch.float32).to(self.device).squeeze(-1)

                z_mu, z_cov = get_stat_Z(Z)
                Z_sample = sample_Z(z_mu, z_cov, bs).view(-1, self.num_markers, self.num_joints, 3)

                X = LBS(self.w, Y, Z_sample)

                beta = 0.5 / (self.conv_to_m * avg_bone)
                X_hat = corrupt(X, beta=beta)
                Z_pw = preweighted_Z(self.w, Z_sample)

                X = normalize_X(X_hat, X)
                Z = normalize_Z_pw(Z_pw)

                Y_hat = self.model(X, Z).view(bs, self.num_joints, 3, 4)
                Y_hat = denormalize_Y(Y_hat, Y)

                loss = self.user_weights * self.criterion(Y_hat, Y)
                total_loss += loss.item() #/ (self.num_joints * 3 * 4)
                angle_diff, translation_diff = transformation_diff(Y_hat, Y)
                total_angle_diff += torch.sum(torch.abs(angle_diff), axis=0)
                total_translation_loss += torch.sum(translation_diff, axis=0)

                tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_translation_loss /= n
        total_angle_diff /= n
        self.val_writer.add_scalar('Loss', total_loss, epoch)
        for i in range(self.num_joints):
            self.val_writer.add_scalar(f'joint_{i+1}: avg angle diff', (total_angle_diff[i] / np.pi) * 180, epoch)
            self.val_writer.add_scalar(f'joint_{i+1}: avg translation diff', total_translation_loss[i], epoch)

        tqdm_update = "Val: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        message = f"epoch: {epoch}, loss: {total_loss}"
        tqdm_batch.close()
        return total_loss, message

    def test_one_animation(self):
        self.test_dataset = MoCap(csv_file=self.cfg.csv_file , fnames=self.cfg.test_filenames,\
                                num_marker=self.num_markers, num_joint=self.num_joints, test=True)

        self.test_steps = len(self.test_dataset)

        self.test_data_loader = DataLoader(self.test_dataset, batch_size=1, \
                                            shuffle=False, num_workers=8)

        self.build_model()
        self.model.to(self.device)
        self.criterion = self.build_loss_function()

        if os.path.exists(self.checkpoint_dir):
            self.load_model()
        else:
            print("There is no saved model")
            exit()

        self.model.eval()
        for batch_idx, (Y, Z, F, avg_bone) in enumerate(self.test_data_loader):
            bs = Y.shape[1]
            Y, Z, F, avg_bone = Y.to(torch.float32).to(self.device).squeeze(0), Z.to(torch.float32).to(self.device).squeeze(0),\
                                F.to(torch.float32).to(self.device).squeeze(0).unsqueeze(1), avg_bone.to(torch.float32).to(self.device).squeeze(-1)

            z_mu, z_cov = get_stat_Z(Z)
            Z_sample = sample_Z(z_mu, z_cov, bs).view(-1, self.num_markers, self.num_joints, 3)

            X = LBS(self.w, Y, Z_sample)

            beta = 0.5 / (self.conv_to_m * avg_bone)
            X_hat = corrupt(X, beta=beta)
            Z_pw = preweighted_Z(self.w, Z_sample)

            X = normalize_X(X_hat, X)
            Z = normalize_Z_pw(Z_pw)

            Y_hat = self.model(X, Z).view(bs, self.num_joints, 3, 4)
            Y_hat = denormalize_Y(Y_hat, Y)

            loss = self.user_weights * self.criterion(Y_hat, Y) / bs

            Y_hat_4x4 = xform_to_mat44_torch(Y_hat)
            Y_ = F @ Y_hat_4x4
            Y_ = Y_.cpu().detach().numpy()
            np.save(f"asd.npy", Y_)
            print("loss={0:.4f}".format(loss.item()))

    def ls_solver(self):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True)
        self.build_model()
        self.criterion = self.build_loss_function()
        self.load_data()

        total_loss = 0
        for batch_idx, (X, Y, Z, F, avg_bone) in enumerate(self.val_data_loader):
            X, Y, Z, F, avg_bone = X.to(torch.float32), Y.to(torch.float32),\
                        Z.to(torch.float32), F.to(torch.float32), avg_bone.to(torch.float32)
            X, Y, Z, F, avg_bone = X.to(self.device), Y.to(self.device), Z.to(self.device), F.to(self.device), avg_bone.to(self.device).squeeze(-1)

            # TODO:
            # do preprocessing and get corrupted X and preweighted Z
            Y_hat = self.model(X, Z)

            loss = self.user_weights * self.criterion(Y_hat, Y)
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
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.AmsGrad.lr,
                                    weight_decay=self.cfg.optimizer.AmsGrad.weight_decay, amsgrad=True)

    def build_loss_function(self):
        return nn.L1Loss(reduction='sum')

    def save_model(self, epoch):
        ckpt = {'model': self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model'])
        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']
