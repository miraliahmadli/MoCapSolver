import torch
import torch.nn as nn
from agents.base_agent import BaseAgent


class MS_Agent(BaseAgent):
    def __init__(self, cfg, test=False, sweep=False):
        super(RS_Agent, self).__init__(cfg, test)
        pass

    def build_model(self,):
        pass

    def load_data(self, ):
        pass

    def train(self,):
        pass

    def run_batch(self, ):
        pass

    def train_one_epoch(self, ):
        pass

    def val_one_epoch(self,):
        pass

    def test_one_animation(self,):
        pass

    def build_loss_function(self):
        pass
