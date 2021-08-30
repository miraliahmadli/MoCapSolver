import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from models.mocap_solver import AE, MocapSolver, Decoder, AE_loss, MS_loss
from datasets.mocap_solver import MS_Dataset


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(RS_Agent, self).__init__(cfg, test)
        self.checkpoint_dir_ae = cfg.autoencoder_dir
        self.checkpoint_dir_ms = cfg.mocap_solver_dir

        self.joint_weights = cfg.joint_weights
        self.skinning_w = cfg.weight_assignment

        self.betas = cfg.loss.autoencoder.betas
        self.alphas = cfg.loss.autoencoder.alphas

        self.joint_topology = self.get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.auto_encoder = AE(edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
                                offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])
        self.ms_model = MocapSolver(self.num_markers, self.cfg.window_size, 1024,
                                            use_motion=True, use_marker_conf=True, use_skeleton=True)
        self.ms_decoder = self.auto_encoder.decoder

    def load_data(self):
        self.train_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.cfg.batch_size
        self.val_steps = len(self.val_dataset) // self.cfg.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        pass

    def run_batch_ae(self, X_c, X_t, X_m, Y_c, Y_t, Y_m):
        l_c, l_t, l_m, Y_hat_c, Y_hat_t, Y_hat_m = self.auto_encoder(X_c, X_t, X_m)

        loss = self.crit_ar((Y_c, Y_t, Y_m), (Y_hat_c, Y_hat_t, Y_hat_m))
        return loss

    def run_batch_ms(self, X, Y, Y_c, Y_t, Y_m):
        l_c, l_t, l_m = self.ms_model(X)
        l_m = l_m.view(l_m.shape[0], 16, -1)
        Y_hat_c, Y_hat_t, Y_hat_m = self.ms_decoder(l_c, l_t, l_m)

        loss = self.crit_ms((Y, Y_c, Y_t, Y_m), (X, Y_hat_c, Y_hat_t, Y_hat_m))
        return loss

    def train_one_epoch(self, epoch):
        pass

    def val_one_epoch(self, epoch):
        pass

    def test_one_animation(self):
        pass

    def build_loss_function(self, ae=True):
        if ae:
            return AE_loss(self.joint_weights, self.betas, self.skinning_w)
        else:
            return MS_loss(self.joint_weights, self.alphas)

    def save_model(self, epoch, ckpt_dir, ae=True):
        if ae:
            ckpt = {'encoder': self.auto_encoder.encoder.state_dict(),
                    'decoder': self.auto_encoder.decoder.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    "epoch": epoch}
        else:
            ckpt = {'mocap_solver': self.ms_model.state_dict(),
                    'decoder': self.ms_decoder.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    "epoch": epoch}
        torch.save(ckpt, ckpt_dir)

    def load_model(self, ckpt_dir, ae=True):
        ckpt = torch.load(ckpt_dir)
        if ae:
            self.auto_encoder.encoder.load_state_dict(ckpt["encoder"])
            self.auto_encoder/Decoder.load_state_dict(ckpt['decoder'])
        else:
            self.ms_model.load_state_dict(ckpt['mocap_solver'])
            self.ms_decoder.load_state_dict(ckpt['decoder'])

        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']
