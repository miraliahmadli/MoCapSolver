import os
import sys
import json
import time
import argparse
from easydict import EasyDict
import wandb

from train import Agent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                         default="train", help="training, testing or sweep")
    parser.add_argument('--config', type=str,
                         default="configs/config.json", 
                         help="path to the config file")
    parser.add_argument('--sweep-config', type=str,
                         default="configs/sweep_config.json",
                         help="wandb.ai sweep config")

    args = parser.parse_args()
    return args


def read_cfg(cfg_file):
    with open(cfg_file) as f:
      cfg = json.loads(f.read())
    cfg = EasyDict(cfg)
    return cfg


def main():
    args = parse_arguments()
    cfg = read_cfg(args.config)
    if args.mode == 'train':
        agent = Agent(cfg)
        agent.train()
    elif args.mode == "test":
        agent = Agent(cfg, True)
        agent.test_one_animation()
    elif args.mode == "sweep":
        with open(args.sweep_config) as f:
            sweep_cfg = json.loads(f.read())
        sweep_id = wandb.sweep(sweep_cfg, project='denoising', entity='mocap')
        cfg.sweep_id = sweep_id
        agent = Agent(cfg, sweep=True)
        wandb.agent(sweep_id, agent.train)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
    # import numpy as np
    # import torch
    # from tools.preprocess import clean_XYZ_torch
    # x = np.random.rand(4, 41, 1)
    # y = np.random.rand(1, 31, 3, 4)
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    # x, y, z = clean_XYZ_torch(x, y, 2.5)
