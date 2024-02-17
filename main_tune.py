import argparse
import sys
import os
import torch
import time
from experiment.tune_and_exp import tune_and_experiment_multiple_runs
from utils.utils import Logger
from types import SimpleNamespace
from experiment.tune_config import config_default


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Hyper-parameter tuning')
    parser.add_argument('--data', dest='data', default='wisdm', type=str)
    parser.add_argument('--encoder', dest='encoder', default='CNN', type=str)
    parser.add_argument('--agent', dest='agent', default='DT2W', type=str)
    parser.add_argument('--norm', dest='norm', default='BN', type=str)
    args = parser.parse_args()

    # Include unchanged general params
    args = SimpleNamespace(**vars(args), **config_default)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_1 = args.encoder + '_' + args.data
    exp_path_2 = args.agent + '_' + args.norm + '_' + exp_start_time
    exp_path = os.path.join(args.path_prefix, exp_path_1, exp_path_2)  # Path for running the whole experiment
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    args.exp_path = exp_path
    log_path = args.exp_path + '/log.txt'
    sys.stdout = Logger('{}'.format(log_path))

    tune_and_experiment_multiple_runs(args)
