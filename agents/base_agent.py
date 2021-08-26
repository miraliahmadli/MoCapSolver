import torch
import wandb
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.is_test = test
        self.checkpoint_dir = cfg.model_dir

        self.num_markers = cfg.num_markers
        self.num_joints = cfg.num_joints
        self.batch_size = cfg.batch_size

        if self.cfg.use_gpu and torch.cuda.is_available():
            self.device =  torch.device(f'cuda:{self.cfg.gpu_idx}')
        else:
            self.device = torch.device("cpu")
        print(self.device)
        if self.cfg.use_gpu and torch.cuda.is_available():
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

        self.model = None
        self.optimizer = None
        self.best_loss = float("inf")

    def build_model(self):
        pass

    def load_data(self):
        pass

    def train(self):
        pass

    def run_batch(self):
        pass

    def train_one_epoch(self):
        pass

    def val_one_epoch(self):
        pass

    def test_one_animation(self):
        pass

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
        pass

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

    def write_summary(self, summary_writer, losses, ang_diff, tr_diff, epoch):
        total_loss, total_loss_rot, total_loss_tr = losses
        summary_writer.add_scalar('Loss', total_loss, epoch)
        summary_writer.add_scalar('Rotation Loss', total_loss_rot, epoch)
        summary_writer.add_scalar('Translation Loss', total_loss_tr, epoch)
        for i in range(self.num_joints):
            summary_writer.add_scalar(f'joint_{i+1}: avg angle diff', ang_diff[i], epoch)
            summary_writer.add_scalar(f'joint_{i+1}: avg translation diff', tr_diff[i], epoch)
        summary_writer.add_scalar('Rotational Error (deg)', torch.mean(ang_diff), epoch)
        summary_writer.add_scalar('Translation Error (mm)', torch.mean(tr_diff), epoch)

    def default_cfg(self):
        return {
            "lr": self.cfg.optimizer.AmsGrad.lr,
            "decay": self.cfg.lr_scheduler.ExponentialLR.decay
        }

    def wandb_summary(self, training, losses, ang_diff, tr_diff, epoch):
        total_loss, total_loss_rot, total_loss_tr = losses
        if not training:
            wandb.log({'Validation Loss': total_loss, 'Epoch': epoch})
            wandb.log({'Validation Rotation Loss': total_loss_rot, 'Epoch': epoch})
            wandb.log({'Validation Translation Loss': total_loss_tr, 'Epoch': epoch})
            for i in range(self.num_joints):
                wandb.log({f'Validation joint_{i+1}: Avg rotation error': ang_diff[i], 'Epoch': epoch})
                wandb.log({f'Validation joint_{i+1}: Avg translation error': tr_diff[i], 'Epoch': epoch})
            wandb.log({'Validation Rotation Error (deg)': torch.mean(ang_diff), 'Epoch': epoch})
            wandb.log({'Validation Translation Error (mm)': torch.mean(tr_diff), 'Epoch': epoch})
        else:
            wandb.log({'Training Loss': total_loss, 'Epoch': epoch})
            wandb.log({'Training Rotation Loss': total_loss_rot, 'Epoch': epoch})
            wandb.log({'Training Translation Loss': total_loss_tr, 'Epoch': epoch})
            for i in range(self.num_joints):
                wandb.log({f'Training joint_{i+1}: Avg rotation error': ang_diff[i], 'Epoch': epoch})
                wandb.log({f'Training joint_{i+1}: Avg translation error': tr_diff[i], 'Epoch': epoch})
            wandb.log({'Training Rotation Error (deg)': torch.mean(ang_diff), 'Epoch': epoch})
            wandb.log({'Training Translation Error (mm)': torch.mean(tr_diff), 'Epoch': epoch})
