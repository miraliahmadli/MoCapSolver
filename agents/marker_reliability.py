import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from models import MarkerReliability
from datasets.marker_reliability import MR_Dataset


class MR_Agent(BaseAgent):
    def __init__(self, cfg, sweep=False):
        super(MR_Agent, self).__init__(cfg, False, False)
        self.reference_markers = cfg.reference_markers

    def build_model(self):
        self.model = MarkerReliability(self.num_markers, len(self.reference_markers))

    def load_data(self):
        self.train_dataset = MR_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.train_filenames,
                                        num_marker=self.num_markers, reference_markers=self.reference_markers)
        self.val_dataset = MR_Dataset(csv_file=self.cfg.csv_file , file_stems=self.cfg.val_filenames,
                                      num_marker=self.num_markers, reference_markers=self.reference_markers)

        self.train_steps = len(self.train_dataset) // self.batch_size
        self.val_steps = len(self.val_dataset) // self.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,\
                                            shuffle=True, num_workers=8, pin_memory=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,\
                                            shuffle=False, num_workers=8, pin_memory=True)

    def train_one_epoch(self, epoch):
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

            tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_loss /= n
        self.write_summary(self.train_writer, total_loss, epoch)
        self.wandb_summary(True, total_loss, epoch)

        tqdm_update = "Train: Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def val_one_epoch(self):
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

                tqdm_update = "Epoch={0:04d},loss={1:.4f}".format(epoch, loss.item() / bs)
                tqdm_batch.set_postfix_str(tqdm_update)
                tqdm_batch.update()

        total_loss /= n
        self.write_summary(self.val_writer, total_loss, epoch)
        self.wandb_summary(False, total_loss, epoch)

        tqdm_update = "Val  : Epoch={0:04d},loss={1:.4f}".format(epoch, total_loss)
        tqdm_batch.set_postfix_str(tqdm_update)
        tqdm_batch.update()
        tqdm_batch.close()

        message = f"epoch: {epoch}, loss: {total_loss}"
        return total_loss, message

    def build_loss_function(self):
        return nn.CrossEntropyLoss()

    def write_summary(self, summary_writer, total_loss, epoch):
        summary_writer.add_scalar('Loss', total_loss, epoch)

    def wandb_summary(self, training, total_loss, epoch):
        if not training:
            wandb.log({'Validation Loss': total_loss, 'Epoch': epoch})
        else:
            wandb.log({'Training Loss': total_loss, 'Epoch': epoch})
