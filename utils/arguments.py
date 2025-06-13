import argparse
import yaml
from yaml import Loader
import torch as T 
import numpy as np 

def read_agrs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='env seed')
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--use_wandb', default=False, action = 'store_true')
    parser.add_argument('--output_dir', default='data/output', type=str)
    parser.add_argument('--show_online', default=False, action = 'store_true')
    parser.add_argument('--use_cpu', default=False, action='store_true')
    parser.add_argument('--learning_stage', default=3, type=int)
    parser.add_argument('--use_matrix', default=True, action='store_false')
    parser.add_argument('--algo', default='iql', type=str) # dqn / a_cmommt
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--full_test', default=False, action='store_true')
    parser.add_argument('--use_rnn', default=False, action='store_true')
    parser.add_argument("--params", nargs='*', help="其他参数，格式为 key=value") # 用于传递其他参数
    parser.add_argument('--use_acmommt', default=False, action='store_true') # 是否使用acmommt
    parser.add_argument('--use_pamts', default=False, action='store_true') # 是否使用acmommt
    args = parser.parse_args()

    return args