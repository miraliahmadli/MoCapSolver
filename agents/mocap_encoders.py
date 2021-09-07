from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import AE, AE_loss
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
                        offset_channels=[1, 8], offset_joint_num=[self.num_joints, 7]).to(self.device)

    def load_data(self):
        self.train_dataset = AE_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.train_filenames, 
                                        num_marker=self.num_markers, num_joint=self.num_joints)
        self.val_dataset = AE_Dataset(csv_file=self.cfg.datadir , file_stems=self.cfg.val_filenames, 
                                      num_marker=self.num_markers, num_joint=self.num_joints)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train_one_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        n = 0

        self.model.train()
        for batch_idx, (X_c, X_t, X_m) in enumerate(self.train_data_loader):
            bs = X_c.shape[0]
            n += bs
            X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

            _, _, _, Y_hat_c, Y_hat_t, Y_hat_m = self.nodel(X_c, X_t, X_m)
            loss_t, loss_m, loss_c = self.criterion((X_c, X_t, X_m), (Y_hat_c, Y_hat_t, Y_hat_m))
            total_loss_c += loss_c.item()
            total_loss_t += loss_t.item()
            total_loss_m += loss_m.item()

            self.optimizer.zero_grad()
            loss_c.backward()
            loss_t.backward()
            loss_m.backward()
            self.optimizer.step()

            tqdm_update = f"Epoch={epoch:04d},offset_loss={loss_c.item() / bs:.4f}, skeletal_loss={loss_t.item() / bs:.4f}, motion_loss={loss_m.item() / bs:.4f}"
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss_c /= n
        total_loss_t /= n
        total_loss_m /= n
        # self.write_summary(self.train_writer, total_loss, epoch)
        # self.wandb_summary(True, total_loss, epoch)

        # tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        # tqdm_batch.set_postfix_str(tqdm_update)
        # tqdm_batch.update()
        # tqdm_batch.close()

        # message = f"epoch: {epoch}, loss: {total_loss}"
        # return total_loss, message

    def val_one_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss_c = 0
        total_loss_t = 0
        total_loss_m = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (X_c, X_t, X_m) in enumerate(self.val_data_loader):
                bs = X_c.shape[0]
                n += bs
                X_c, X_t, X_m = X_c.to(torch.float32).to(self.device), X_t.to(torch.float32).to(self.device),  X_m.to(torch.float32).to(self.device)

                _, _, _, Y_hat_c, Y_hat_t, Y_hat_m = self.nodel(X_c, X_t, X_m)
                loss_c, loss_t, loss_m = self.criterion((X_c, X_t, X_m), (Y_hat_c, Y_hat_t, Y_hat_m))
                total_loss_c += loss_c.item()
                total_loss_t += loss_t.item()
                total_loss_m += loss_m.item()

                tqdm_update = f"Epoch={epoch:04d},offset_loss={loss_c.item() / bs:.4f}, skeletal_loss={loss_t.item() / bs:.4f}, motion_loss={loss_m.item() / bs:.4f}"
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

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
