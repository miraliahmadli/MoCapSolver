import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from models import AE, AE_loss, MarkerReliability
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import AE_Dataset


class EncoderAgent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(RS_Agent, self).__init__(cfg, test, sweep)
        self.joint_weights = cfg.joint_weights
        self.marker_weights = cfg.marker_weights
        self.skinning_w = cfg.weight_assignment

        self.betas = cfg.loss.autoencoder.betas
        self.alphas = cfg.loss.autoencoder.alphas

        self.joint_topology = self.get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.nodel = AE(edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
                                offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])
        self.f_mr = MarkerReliability(self.num_markers, 8)

    def load_data(self):
        self.train_dataset = AE_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = AE_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,\
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        pass

    def run_batch(self, X_c, X_t, X_m, Y_c, Y_t, Y_m):
        l_c, l_t, l_m, Y_hat_c, Y_hat_t, Y_hat_m = self.nodel(X_c, X_t, X_m)

        loss = self.criterion((Y_c, Y_t, Y_m), (Y_hat_c, Y_hat_t, Y_hat_m))
        return loss

    def train_one_epoch(self, epoch):
        pass

    def val_one_epoch(self, epoch):
        pass

    def test_one_animation(self):
        pass

    def build_loss_function(self):
        return AE_loss(self.marker_weights , self.joint_weights, self.betas, self.skinning_w)

    def save_model(self, epoch):
        ckpt = {'encoder': self.nodel.encoder.state_dict(),
                'decoder': self.nodel.decoder.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.nodel.encoder.load_state_dict(ckpt["encoder"])
        self.nodel.decoder.load_state_dict(ckpt['decoder'])

        if not self.is_test:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.best_loss = ckpt['best_loss']

        return ckpt['epoch']

    def load_normalizer(self, ckpt_dir):
        ckpt = torch.load(ckpt_dir)
        self.f_mr.load_state_dict(ckpt['model'])
