import os
import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from models import AE, MocapSolver, MarkerReliability, MS_loss
from models.mocap_solver import Decoder
from models.mocap_solver.skeleton import get_topology, build_edge_topology
from datasets.mocap_solver import MS_Dataset


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(RS_Agent, self).__init__(cfg, test, sweep)
        self.joint_weights = cfg.joint_weights
        self.skinning_w = cfg.weight_assignment

        self.betas = cfg.loss.autoencoder.betas
        self.alphas = cfg.loss.autoencoder.alphas

        self.joint_topology = self.get_topology(cfg.hierarchy, self.num_joints)
        self.edges = build_edge_topology(self.joint_topology, torch.zeros((self.num_joints, 3)))

    def build_model(self):
        self.auto_encoder = AE(self.edges, self.num_markers, self.num_joints, 1024, offset_dims=self.cfg.model.ae.offset_dims, 
                                offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7])
        self.model = MocapSolver(self.num_markers, self.cfg.window_size, 1024,
                                use_motion=True, use_marker_conf=True, use_skeleton=True)
        self.ms_decoder = self.auto_encoder.decoder
        if os.path.exists(self.cfg.model.decoder_dir):
            self.load_decoder(self.cfg.model.decoder_dir)
        else:
            print("No pretrained encoder")
            exit()
        self.f_mr = MarkerReliability(self.num_markers, 8)

    def load_data(self):
        self.train_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.train_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,
                                num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.val_filenames, lrf_mean_markers_file=self.cfg.lrf_mean_markers,
                                num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        pass

    def run_batch(self, X, Y, Y_c, Y_t, Y_m):
        l_c, l_t, l_m = self.model(X)
        l_m = l_m.view(l_m.shape[0], 16, -1)
        Y_hat_c, Y_hat_t, Y_hat_m = self.ms_decoder(l_c, l_t, l_m)

        loss = self.criterion((Y, Y_c, Y_t, Y_m), (X, Y_hat_c, Y_hat_t, Y_hat_m))
        return loss

    def train_one_epoch(self, epoch):
        pass

    def val_one_epoch(self, epoch):
        pass

    def test_one_animation(self):
        pass

    def build_loss_function(self):
        return MS_loss(self.joint_weights, self.alphas)

    def load_decoder(self, decoder_dir):
        ckpt = torch.load(decoder_dir)
        self.ms_decoder.load_state_dict(ckpt['decoder'])
        self.ms_decoder.freeze_params() # freeze params to avoid backprop

    def load_normalizer(self, ckpt_dir):
        ckpt = torch.load(ckpt_dir)
        self.f_mr.load_state_dict(ckpt['model'])
