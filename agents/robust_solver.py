import os
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import Baseline, LS_solver, RS_loss, VNHoldenModel
from datasets.robust_solver import RS_Dataset, RS_Test_Dataset, RS_Synthetic, RS_Synthetic_Test

from tools.utils import LBS, corrupt, preweighted_Z, xform_to_mat44, symmetric_orthogonalization
from tools.utils import svd_rot as svd_solver
from tools.preprocess import weight_assign
from tools.statistics import *
from tools.transform import transformation_diff


class RS_Agent(BaseAgent):
    def __init__(self, cfg, test=False):
        super(RS_Agent, self).__init__(cfg, test)

        self.use_vn = self.cfg.model.used.lower()
        self.conv_to_m = cfg.scale_factor #0.57803

        self.user_weights_rot = cfg.user_weights_rotation
        self.user_weights_t = cfg.user_weights_translation
        self.w = weight_assign(cfg.weight_assignment, cfg.num_markers, cfg.num_joints)

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
        hidden_size = self.cfg.model.hidden_size
        use_svd = self.cfg.model.use_svd
        num_layers = self.cfg.model.num_layers
        if used == "baseline":
            self.model = Baseline(self.num_markers, self.num_joints, hidden_size, num_layers, use_svd)
        elif used == "vn":
            self.model = VNHoldenModel(self.num_markers, self.num_joints, hidden_size, num_layers, use_svd)
        elif used == "least_square":
            w = weight_assign('dataset/joint_to_marker_three2one.txt').to(self.device)
            self.model = LS_solver(self.num_joints, w)
        else:
            raise NotImplementedError

        self.model.to(self.device)

    def load_data(self):
        if self.cfg.dataset == "CMU":
            self.train_dataset = RS_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                    num_marker=self.num_markers, num_joint=self.num_joints)
            self.val_dataset = RS_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                    num_marker=self.num_markers, num_joint=self.num_joints)
        elif self.cfg.dataset == "Synthetic":
            self.train_dataset = RS_Synthetic(data_dir=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                        num_marker=self.num_markers, num_joint=self.num_joints, local_ref_markers=self.cfg.local_ref_markers)
            self.val_dataset = RS_Synthetic(data_dir=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                        num_marker=self.num_markers, num_joint=self.num_joints, local_ref_markers=self.cfg.local_ref_markers)
        else:
            print("Wrong Dataset")
            exit()

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def run_batch(self, Y, Z, avg_bone, bs, sample_markers, corrupt_markers):
        if sample_markers:
            Z = self.sampler.sample((bs, )).view(-1, self.num_markers, self.num_joints, 3)

        X = LBS(self.w, Y, Z)
        if corrupt_markers:
            beta = 3 / (self.conv_to_m * avg_bone)
            X_hat = corrupt(X, beta=beta.view(-1))
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
    
    def run_batch_vn(self, Y, Z, avg_bone, bs, sample_markers, corrupt_markers):
        if sample_markers:
            Z = self.sampler.sample((bs, )).view(-1, self.num_markers, self.num_joints, 3)

        X = LBS(self.w, Y, Z)
        if corrupt_markers:
            beta = 3 / (self.conv_to_m * avg_bone)
            X_hat = corrupt(X, beta=beta.view(-1))
        else:
            X_hat = X

        root_tr = Y[:, 0:1, :, -1]
        X_hat -= root_tr
        Y_hat = self.model(X_hat)
        Y_hat += root_tr[..., None]

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

            if self.cfg.model.used.lower() == "vn":
                Y_hat = self.run_batch_vn(Y, Z, avg_bone, bs, sample_markers=False, 
                                            corrupt_markers=self.cfg.training_settings.train_set.corrupt)
            else:
                Y_hat = self.run_batch(Y, Z, avg_bone, bs,\
                                        sample_markers=self.cfg.training_settings.train_set.sample,
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
            
            Y_h_orth = Y_h.reshape(-1, 3, 4)
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

                if self.cfg.model.used.lower() == "vn":
                    Y_hat = self.run_batch_vn(Y, Z, avg_bone, bs, sample_markers=False, 
                                                corrupt_markers=self.cfg.training_settings.train_set.corrupt)
                else:
                    Y_hat = self.run_batch(Y, Z, avg_bone, bs,\
                                            sample_markers=self.cfg.training_settings.train_set.sample,\
                                            corrupt_markers=self.cfg.training_settings.train_set.corrupt)

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

                Y_h_orth = Y_h.reshape(-1, 3, 4)
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
        if self.cfg.dataset == "CMU":
            self.test_dataset = RS_Test_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.test_filenames,\
                                num_marker=self.num_markers, num_joint=self.num_joints)
        elif self.cfg.dataset == "Synthetic":
            self.test_dataset = RS_Synthetic_Test(data_dir=self.cfg.test_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                        num_marker=self.num_markers, num_joint=self.num_joints, local_ref_markers=self.cfg.local_ref_markers)
        else:
            print("Wrong Dataset name")
            exit()
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

                Y_hat_4x4 = xform_to_mat44(Y_hat)
                Y_ = F @ Y_hat_4x4
                Y_ = Y_.cpu().detach().numpy()

                Y_4x4 = xform_to_mat44(Y)
                Y_gt = F @ Y_4x4
                Y_gt = Y_gt.cpu().detach().numpy()
                
                # X = X.cpu().detach().numpy()
                all_Y = np.concatenate((Y_gt[None, :], Y_[None, :]), axis=0)

                from tools.viz import visualize
                visualize(Ys=all_Y[..., [1, 2, 0], :])#, Xs=X, fps_vid=120.0)
                exit()
                # np.save(f"asd.npy", Y_)
                print("loss={0:.4f}".format(loss.item()))

    def build_loss_function(self):
        return RS_loss(self.user_weights_t, self.user_weights_rot)

    def default_cfg(self):
        return {
            "use_svd": self.cfg.model.use_svd,
            "lr": self.cfg.optimizer.AmsGrad.lr,
            "decay": self.cfg.lr_scheduler.ExponentialLR.decay
        }
