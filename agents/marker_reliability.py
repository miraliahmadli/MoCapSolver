from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent
from models import MarkerReliability
from datasets.marker_reliability import MR_Dataset


class MR_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(MR_Agent, self).__init__(cfg, False, False)

    def build_model(self):
        self.model = MarkerReliability(self.num_markers, 8, hidden_size=self.cfg.model.hidden_size, 
                                window_size=self.cfg.window_size, num_res_layers=self.cfg.model.num_layers).to(self.device)

    def load_data(self):
        self.train_dataset = MR_Dataset(data_dir=self.cfg.train_filenames, window_size=self.cfg.window_size, threshold=self.cfg.threshold)
        self.val_dataset = MR_Dataset(data_dir=self.cfg.val_filenames, window_size=self.cfg.window_size, threshold=self.cfg.threshold)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True) 
        total_loss = 0
        n = 0

        self.model.train()
        for batch_idx, (raw_marker, rel_scores) in enumerate(self.train_data_loader):
            bs = raw_marker.shape[0]
            n += bs
            raw_marker, rel_scores = raw_marker.to(torch.float32).to(self.device), rel_scores.to(torch.float32).to(self.device)
            pred_rel_scores = self.model(raw_marker)
            loss = self.criterion(pred_rel_scores, rel_scores)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / (bs * self.cfg.window_size))
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= (n * self.cfg.window_size)
        self.write_summary(self.train_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, epoch)

        tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_per_epoch(self, epoch):
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True) 
        total_loss = 0
        n = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (raw_marker, rel_scores) in enumerate(self.val_data_loader):
                bs = raw_marker.shape[0]
                n += bs
                raw_marker, rel_scores = raw_marker.to(torch.float32).to(self.device), rel_scores.to(torch.float32).to(self.device)
                pred_rel_scores = self.model(raw_marker)
                loss = self.criterion(pred_rel_scores, rel_scores)
                total_loss += loss.item()

                tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / (bs * self.cfg.window_size))
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= (n * self.cfg.window_size)
        self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, epoch)

        tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def build_loss_function(self):
        return nn.BCELoss(reduction="sum")

    def write_summary(self, summary_writer, total_loss, epoch):
        summary_writer.add_scalar('Loss', total_loss, epoch)

    def wandb_summary(self, training, total_loss, epoch):
        if not training:
            wandb.log({'Validation Loss': total_loss, 'Epoch': epoch})
        else:
            wandb.log({'Training Loss': total_loss, 'Epoch': epoch})
