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
        self.train_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.train_filenames,
                                        num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = MS_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.val_filenames,
                                      num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train(self):
        pass

    def normalize(self, X):
        P = self.f_mr(X)
        if not (P < 0.8).any(): # reliable
            pass
        else:
            pass

    def run_batch(self, X, Y, Y_c, Y_t, Y_m):
        X = self.normalize(X)
        l_c, l_t, l_m = self.model(X)
        l_m = l_m.view(l_m.shape[0], 16, -1)
        Y_hat_c, Y_hat_t, Y_hat_m = self.ms_decoder(l_c, l_t, l_m)

        losses = self.criterion((Y, Y_c, Y_t, Y_m), (X, Y_hat_c, Y_hat_t, Y_hat_m))
        return losses # loss, loss_marker, loss_c, loss_t, loss_m

    def train_one_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        n = 0

        self.model.train()
        for batch_idx, (X, Y, Y_c, Y_t, Y_m) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X, Y = X.to(torch.float32).to(self.device), Y.to(torch.float32).to(self.device)
            Y_c, Y_t, Y_m = Y_c.to(torch.float32).to(self.device), Y_t.to(torch.float32).to(self.device),  Y_m.to(torch.float32).to(self.device)

            loss, loss_marker, loss_c, loss_t, loss_m = self.run_batch(X, Y, Y_c, Y_t, Y_m)
            total_loss += loss.item()
            total_loss_c += loss_c.item()
            total_loss_t += loss_t.item()
            total_loss_m += loss_m.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        total_loss_c /= n
        total_loss_t /= n
        total_loss_m /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_batch.set_postfix_str(tqdm_update)
        # tqdm_batch.update()
        # tqdm_batch.close()

        # message = f"epoch: {epoch}, loss: {total_loss}"
        # return total_loss, message

    def val_one_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y, Y_c, Y_t, Y_m) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X, Y = X.to(torch.float32).to(self.device), Y.to(torch.float32).to(self.device)
                Y_c, Y_t, Y_m = Y_c.to(torch.float32).to(self.device), Y_t.to(torch.float32).to(self.device),  Y_m.to(torch.float32).to(self.device)

                loss, loss_marker, loss_c, loss_t, loss_m = self.run_batch(X, Y, Y_c, Y_t, Y_m)
                total_loss += loss.item()

                tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        # self.write_summary(self.val_writer, total_loss, epoch)
        # self.wandb_summary(False, total_loss, epoch)

        # tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_batch.set_postfix_str(tqdm_update)
        # tqdm_batch.update()
        # tqdm_batch.close()

        # message = f"epoch: {epoch}, loss: {total_loss}"
        # return total_loss, message

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
