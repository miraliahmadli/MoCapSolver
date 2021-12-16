import os
import sys
import json
import time
import argparse
from easydict import EasyDict

from agents.robust_solver import RS_Agent
from agents.mocap_solver import MS_Agent
from agents.mocap_encoders import EncoderAgent
from agents.marker_reliability import MR_Agent
from agents.ts_encoder import TS_Agent
from agents.mc_encoder import MC_Agent
from agents.motion_encoder import Motion_Agent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                         default="RobustSolver", help="model to train")
    parser.add_argument('--mode', type=str,
                         default="train", help="training, testing or sweep")
    parser.add_argument('--config', type=str,
                         default="configs/config.json", 
                         help="path to the config file")

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
    test = args.mode == "test"
    model = args.model
    if model == "RobustSolver":
        agent = RS_Agent(cfg, test)
    elif model == "MocapSolver":
        agent = MS_Agent(cfg, test)
    elif model == "Autoencoder":
        agent = EncoderAgent(cfg, test)
    elif model == "Normalizer":
        agent = MR_Agent(cfg, test)
    elif model == "TS":
        agent = TS_Agent(cfg, test)
    elif model == "MC":
        agent = MC_Agent(cfg, test)
    elif model == "Motion":
        agent = Motion_Agent(cfg, test)

    if test:
        agent.test_one_animation()
    elif args.mode == 'train':
        agent.train()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
