import os
import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import AE, MocapSolver, MS_loss
from models.mocap_solver import Decoder, MocapSolver, FK, LBS_motion
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import MS_Dataset
from tools.transform import quaternion_to_matrix, matrix_to_quaternion, transformation_diff
from tools.utils import xform_to_mat44


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(MS_Agent, self).__init__(cfg, test, sweep)
        # torch.autograd.set_detect_anomaly(True)
        self.ts_checkpoint_dir = cfg.ts_model
        self.mc_checkpoint_dir = cfg.mc_model
        self.motion_checkpoint_dir = cfg.motion_model
        self.train_ts_decoder = cfg.model.train_ts_decoder
        self.train_mc_decoder = cfg.model.train_mc_decoder
        self.train_motion_decoder = cfg.model.train_motion_decoder
        self.joint_weights = torch.tensor(cfg.joint_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.marker_weights = torch.tensor(cfg.marker_weights, dtype=torch.float32, device=self.device).view(-1, 1)
        self.offset_weights = torch.tensor(np.load(cfg.offset_weights), dtype=torch.float32, device=self.device)
        self.offset_weights = self.offset_weights.view(*self.offset_weights.shape, 1)
        self.skinning_w = torch.tensor(np.load(cfg.weight_assignment), dtype=torch.float32, device=self.device)

        self.alphas = cfg.loss.alphas
        self.window_size = cfg.window_size

        self.joint_topology = get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        # self.auto_encoder = AE(self.edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
        #                        offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])#.to(self.device)
        # self.mocap_solver = MocapSolver(self.num_markers, self.window_size, 1024,
        #                         use_motion=True, use_marker_conf=True, use_skeleton=True)#.to(self.device)
        # self.ms_decoder = self.auto_encoder.decoder.to(self.device)
        # self.ms_decoder = Decoder(self.auto_encoder.encoder)#.to(self.device)
        self.model = MocapSolver(self.edges).to(self.device)
        self.load_pretrained_models()

    def load_data(self):
        self.train_dataset = MS_Dataset(data_dir=self.cfg.train_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)
        self.val_dataset = MS_Dataset(data_dir=self.cfg.val_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames, 
                                        local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=64, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=64, pin_memory=True)

    def smooth_batch(self, X):
        bs = X.shape[0]
        overlap = self.cfg.overlap_frames
        overlap_ = self.window_size - overlap

        frames = bs * self.window_size - (bs - 1)*overlap
        X_res = torch.zeros((frames, *X.shape[2:]), device=X.device, dtype=torch.float32)
        div = [0]*frames
        idx = 0
        b_idx = 0
        while idx < frames and b_idx < bs:
            for i in range(self.window_size):
                X_res[idx + i] += X[b_idx, i]
                div[idx + i] += 1
            idx += overlap_
            b_idx += 1
        for i in range(frames):        
            X_res[i] /= div[i]

        return X_res

    def run_batch(self, X, X_c, X_t, X_m, F, test=False):
        xm_root_quat = X_m[..., :4] # ... x T x 4
        xm_root_quat = nn.functional.normalize(xm_root_quat, dim=-1)
        l_c, l_t, l_m, Y_c, Y_t, Y_m, ym_root_quat = self.model(X)

        bs = X.shape[0]
        lat_t = self.model.static_encoder(X_t.view(bs, -1))
        offset = [X_t, lat_t]
        crit_lat = nn.SmoothL1Loss()
        latent_loss = 0
        if not self.train_ts_decoder:
            latent_loss += 2*crit_lat(l_t, lat_t)
        if not self.train_mc_decoder:
            lat_c = self.model.mc_encoder(X_c.view(bs, -1), offset)
            latent_loss += crit_lat(l_c, lat_c)
        if not self.train_motion_decoder:
            lat_m = self.model.dynamic_encoder(X_m[:, :, 4:].transpose(-1, -2), offset)
            latent_loss += crit_lat(l_m, lat_m)
        
        crit_rot = nn.MSELoss()
        first_rot_loss = 100*crit_rot(ym_root_quat, xm_root_quat)

        # skinning
        j_rot, j_tr = FK(self.joint_topology, Y_m, Y_t)
        j_xform = torch.cat([j_rot, j_tr.unsqueeze(-1)], dim=-1)
        Y = LBS_motion(self.skinning_w, Y_c, j_xform)

        return Y_c, Y_t, Y_m, Y, latent_loss, first_rot_loss

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        total_mp_err = 0
        n = 0

        self.model.mocap_solver.train()
        if self.train_ts_decoder:
            self.model.static_decoder.train()
        if self.train_mc_decoder:
            self.model.mc_decoder.train()
        if self.train_motion_decoder:
            self.model.dynamic_decoder.train()
        for batch_idx, (X_c, X_t, X_m, X, X_clean, F) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X, F, X_clean = X.to(torch.float32).to(self.device), F.to(torch.float32).to(self.device), X_clean.to(torch.float32).to(self.device)
            X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

            Y_c, Y_t, Y_m, Y, latent_loss, first_rot_loss = self.run_batch(X, X_c, X_t, X_m, F)

            losses = self.criterion((Y_c, Y_t, Y_m, Y), (X_c, X_t, X_m, X_clean))
            loss, loss_c, loss_t, loss_m, loss_marker = losses
            loss += latent_loss + first_rot_loss
            total_loss += loss.item() * bs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            glob_r_x, glob_t_x = FK(self.joint_topology, X_m.detach().clone(), X_t.detach().clone())
            glob_r_y, glob_t_y = FK(self.joint_topology, Y_m.detach().clone(), Y_t.detach().clone())
            xform_x = torch.cat([glob_r_x, glob_t_x[..., None]], dim=-1)
            xform_y = torch.cat([glob_r_y, glob_t_y[..., None]], dim=-1)
            angle_diff, translation_diff = transformation_diff(xform_x, xform_y)
            angle_diff = angle_diff / np.pi * 180

            total_angle_err += torch.sum(angle_diff) / (self.window_size * self.num_joints)

            # jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
            # total_jp_err += jp_err.sum() / self.num_joints
            total_jp_err += translation_diff.sum() / (self.window_size * self.num_joints)

            mp_err = torch.norm(X_clean.detach().clone().view(-1, self.num_markers, 3) - Y.detach().clone().view(-1, self.num_markers, 3), 2, dim=-1) # (n x t) x m
            total_mp_err += mp_err.sum() / (self.num_markers * self.window_size)

            tqdm_update = "Epoch={0:04d}, loss={1:.4f}, angle_diff={2:.4f}, jpe={3:.4f}, mpe={4:4f}".format(epoch, 1000*loss.item() / bs, torch.abs(angle_diff).mean(), translation_diff.mean(), mp_err.mean())
            # tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}, loss_marker={5:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item(), loss_marker.item())
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jp_err /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, total_angle_err, total_jp_err, total_mp_err)

        tqdm_update = "Train: Epoch={0:04d}, loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}, mpe={4:4f}".format(epoch, total_loss, total_angle_err, total_jp_err, total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_angle_err = 0
        total_jp_err = 0
        total_mp_err = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m, X, X_clean, F) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X, F, X_clean = X.to(torch.float32).to(self.device), F.to(torch.float32).to(self.device), X_clean.to(torch.float32).to(self.device)
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

                Y_c, Y_t, Y_m, Y, latent_loss, first_rot_loss = self.run_batch(X, X_c, X_t, X_m, F)
                losses = self.criterion((Y_c, Y_t, Y_m, Y), (X_c, X_t, X_m, X_clean))
                loss, loss_c, loss_t, loss_m, loss_marker = losses
                loss += latent_loss + first_rot_loss
                total_loss += loss.item() * bs

                glob_r_x, glob_t_x = FK(self.joint_topology, X_m.detach().clone(), X_t.detach().clone())
                glob_r_y, glob_t_y = FK(self.joint_topology, Y_m.detach().clone(), Y_t.detach().clone())
                xform_x = torch.cat([glob_r_x, glob_t_x[..., None]], dim=-1)
                xform_y = torch.cat([glob_r_y, glob_t_y[..., None]], dim=-1)
                angle_diff, translation_diff = transformation_diff(xform_x, xform_y)
                angle_diff = angle_diff / np.pi * 180

                total_angle_err += torch.sum(angle_diff) / (self.window_size * self.num_joints)

                # jp_err = torch.norm(X_t.detach().clone() - Y_t.detach().clone(), 2, dim=-1) # n x j
                # total_jp_err += jp_err.sum() / self.num_joints
                total_jp_err += translation_diff.sum() / (self.window_size * self.num_joints)

                mp_err = torch.norm(X_clean.detach().clone().view(-1, self.num_markers, 3) - Y.detach().clone().view(-1, self.num_markers, 3), 2, dim=-1) # (n x t) x m
                total_mp_err += mp_err.sum() / (self.num_markers * self.window_size)

                tqdm_update = "Epoch={0:04d}, loss={1:.4f}, angle_diff={2:.4f}, mpe={4:4f}".format(epoch, loss.item(), torch.abs(angle_diff).mean(), translation_diff.mean(), mp_err.mean())
                # tqdm_update = "Epoch={0:04d}, loss={1:.4f}, loss_c={2:.4f}, loss_t={3:.4f}, loss_m={4:4f}, loss_marker={5:4f}".format(epoch, loss.item(), loss_c.item(), loss_t.item(), loss_m.item(), loss_marker.item())
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        total_angle_err /= n
        total_jp_err /= n
        total_mp_err /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, total_angle_err, total_jp_err, total_mp_err)

        tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}, angle_err={2:4f}, jpe={3:4f}, mpe={4:4f}".format(epoch, total_loss, total_angle_err, total_jp_err, total_mp_err)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def test_one_animation(self):
        self.test_dataset = MS_Dataset(data_dir=self.cfg.test_filenames, window_size=self.window_size, overlap=self.cfg.overlap_frames, 
                                    local_ref_markers=self.cfg.local_ref_markers, local_ref_joint=self.cfg.local_ref_joint)

        self.test_steps = len(self.test_dataset)

        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, \
                                            shuffle=False, num_workers=8)

        self.build_model()
        self.model.to(self.device)
        self.criterion = self.build_loss_function()

        if os.path.exists(self.checkpoint_dir):
            epoch = self.load_model()
            print(f"Epoch: {epoch}")
        else:
            print("There is no saved model")
            exit()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m, X, X_clean, F) in enumerate(self.test_data_loader):
                # bs = X_c.shape[0]
                # n += bs
                X, F, X_clean = X.to(torch.float32).to(self.device), F.to(torch.float32).to(self.device), X_clean.to(torch.float32).to(self.device)
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

                Y_c, Y_t, Y_m, Y, _ = self.run_batch(X, X_c, X_t, X_m, F, test=True)
                
                X_, X_clean_, X_c_, X_t_, X_m_ = X.detach().clone(), X_clean.detach().clone(), X_c.detach().clone(), X_t.detach().clone(), X_m.detach().clone()
                Y_c_, Y_t_, Y_m_, Y_ = Y_c.detach().clone(), Y_t.detach().clone(), Y_m.detach().clone(), Y.detach().clone()

                X_denorm = F[..., :3].unsqueeze(2) @ X_[..., None] + F[..., 3, None].unsqueeze(2)
                X_clean_denorm = F[..., :3].unsqueeze(2) @ X_clean_[..., None] + F[..., 3, None].unsqueeze(2)

                X_denorm = X_denorm.squeeze(-1)
                X_clean_denorm = X_clean_denorm.squeeze(-1)

                Y_denorm = F[..., :3].unsqueeze(2) @ Y_[..., None] + F[..., 3, None].unsqueeze(2)
                Y_denorm = Y_denorm.squeeze(-1)
                
                # from tools.viz import visualize
                # Xs = self.smooth_batch(X_clean_denorm)
                # visualize(Xs=Xs.cpu().numpy()[..., [1, 2, 0]] * 10, focus="marker")
                # exit()

                xm_root_rot = quaternion_to_matrix(X_m_[..., :4])
                xm_root_t = X_m_[..., -3:].unsqueeze(-1)
                xm_root_xform = torch.cat((xm_root_rot, xm_root_t), -1)
                xm_root_44 = xform_to_mat44(xm_root_xform) # N x T x 4 x 4
                xm_root = F @ xm_root_44 # N x T x 3 x 4
                xm_root_quat = matrix_to_quaternion(xm_root[..., :3]) # N x T x 4
                X_m_[..., :4] = xm_root_quat
                X_m_[..., -3:] = xm_root[..., -1]

                ym_root_rot = quaternion_to_matrix(Y_m_[..., :4])
                ym_root_t = Y_m_[..., -3:].unsqueeze(-1)
                ym_root_xform = torch.cat((ym_root_rot, ym_root_t), -1)
                ym_root_44 = xform_to_mat44(ym_root_xform) # N x T x 4 x 4
                ym_root = F @ ym_root_44 # N x T x 3 x 4
                ym_root_quat = matrix_to_quaternion(ym_root[..., :3]) # N x T x 4
                Y_m_[..., :4] = ym_root_quat
                Y_m_[..., -3:] = ym_root[..., -1]

                X_all = self.smooth_batch(Y_denorm)
                X_all_gt = self.smooth_batch(X_clean_denorm)
                # mo_all = self.smooth_batch(Y_m_)
                # mo_all_gt = self.smooth_batch(X_m_)
                glob_r_x, glob_t_x = FK(self.joint_topology, X_m_, X_t_[0].unsqueeze(0))
                glob_r_y, glob_t_y = FK(self.joint_topology, Y_m_, Y_t_.mean(dim=0).unsqueeze(0))
                xform_x = torch.cat([glob_r_x, glob_t_x[..., None]], dim=-1)
                xform_y = torch.cat([glob_r_y, glob_t_y[..., None]], dim=-1)
                xform_x = self.smooth_batch(xform_x)
                xform_y = self.smooth_batch(xform_y)
                Xs = 5 * torch.cat((X_all_gt.unsqueeze(0), X_all.unsqueeze(0)), 0).detach().cpu().numpy()
                Ys = 5 * torch.cat((xform_x.unsqueeze(0), xform_y.unsqueeze(0)), 0).detach().cpu().numpy()
                from tools.viz import visualize
                visualize(Xs=Xs[..., [0, 2, 1]], Ys=Ys[...,[0, 2, 1], :], colors_X=[[0, 255, 0, 255], [0, 0, 255, 255]], colors_Y=[[0, 255, 0, 255], [0, 0, 255, 255]], res=(960, 920), fps_vid=120)
                exit()

    def build_loss_function(self):
        return MS_loss(self.joint_topology, self.joint_weights, self.marker_weights, self.offset_weights, self.alphas)

    def wandb_summary(self, training, total_loss, total_angle_err, total_jp_err, total_mp_err):
        if not training:
            wandb.log({'Validation Loss': total_loss})
            wandb.log({'Validation Angle Error (deg)': total_angle_err})
            wandb.log({'Validation Joint Position Error (mm)': 1000*total_jp_err})
            wandb.log({'Validation Marker Position Error (mm)': 1000*total_mp_err})
        else:
            wandb.log({'Training Loss': total_loss})
            wandb.log({'Training Angle Error (deg)': total_angle_err})
            wandb.log({'Training Joint Position Error (mm)': 1000*total_jp_err})
            wandb.log({'Training Marker Position Error (mm)': 1000*total_mp_err})

    def save_model(self, epoch):
        ckpt = {'mocap_solver': self.model.mocap_solver.state_dict(),
                'static_decoder': self.model.static_decoder.state_dict(),
                'mc_decoder': self.model.mc_decoder.state_dict(),
                'dynamic_decoder': self.model.dynamic_decoder.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.mocap_solver.load_state_dict(ckpt["mocap_solver"])
        if self.train_ts_decoder:
            self.model.static_decoder.load_state_dict(ckpt['static_decoder'])
        if self.train_motion_decoder:
            self.model.dynamic_decoder.load_state_dict(ckpt['dynamic_decoder'])
        if self.train_mc_decoder:
            self.model.mc_decoder.load_state_dict(ckpt['mc_decoder'])

        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']

    def load_pretrained_models(self):
        ckpt_ts = torch.load(self.ts_checkpoint_dir)
        self.model.static_encoder.load_state_dict(ckpt_ts["encoder"])
        self.model.static_encoder.eval()

        ckpt_mc = torch.load(self.mc_checkpoint_dir)
        self.model.mc_encoder.load_state_dict(ckpt_mc["encoder"])
        self.model.mc_encoder.eval()

        ckpt_motion = torch.load(self.motion_checkpoint_dir)
        self.model.dynamic_encoder.load_state_dict(ckpt_motion["encoder"])
        self.model.dynamic_encoder.eval()

        if not self.train_ts_decoder:
            self.model.static_decoder.load_state_dict(ckpt_ts["decoder"])
            print(f"Loading TS Model: Epoch #{ckpt_ts['epoch']}")
            self.model.static_decoder.eval()

        if not self.train_mc_decoder:
            self.model.mc_decoder.load_state_dict(ckpt_mc["decoder"])
            print(f"Loading MC Model: Epoch #{ckpt_mc['epoch']}")
            self.model.mc_decoder.eval()

        if not self.train_motion_decoder:
            self.model.dynamic_decoder.load_state_dict(ckpt_motion["decoder"])
            print(f"Loading Motion Model: Epoch #{ckpt_motion['epoch']}")
            self.model.dynamic_decoder.eval()
