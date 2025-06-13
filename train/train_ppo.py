import os
import torch as T
import shutil
from utils.log import init_log
from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
from env_wrapper.popenv import POPEnv
from algos.runner.ppo.ppo_runner import PPORunner
from utils.util import set_all_config
import wandb
from utils.util import set_seed
from utils.log import log_info
from utils.util import set_parallel_env
import json

def train_mappo(args, run_dir, config_dir):
    
    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, f'{args.algo}.yml')
    device = T.device('cuda:0' if T.cuda.is_available() and not args.use_cpu else 'cpu')
    # parameter setting
    logger, save_path, log_file = init_log(run_dir)
    set_seed(args.seed)
    # copy config 
    shutil.copy(train_config_path, save_path)
    shutil.copy(common_config_path, save_path)
    all_config = set_all_config(common_config_path, train_config_path, args, save_path)
    all_config.run_dir = run_dir
    all_config.save_path = save_path
    all_config.device = device
    # set env
    train_env = set_parallel_env(all_config, env_config, env_num=all_config.n_rollout_threads)
    eval_env = set_parallel_env(all_config, env_config, env_num=all_config.n_eval_rollout_threads)
    env = POPEnv(BaseEnv(env_config), all_config)
    env_info = env.get_info()
    all_config.env_info = env_info
    del env
    log_info(logger, all_config, env_info, args)
    env_config_save_pth = os.path.join(all_config.save_path, 'env_config.json')
    with open(env_config_save_pth, 'w')  as f:
        json.dump(env_config.__dict__, f, indent=2)
    from network.ac.ac import PPOAC
    policy = PPOAC(all_config, env_info, device)
    runner = PPORunner(all_config, train_env, policy, eval_env=eval_env)
    runner.logger = logger
    # init_wanb
    if args.use_wandb:
        wandb.init(project="ShadowClone", entity="the-one", group='CMOMMT4-1', config=all_config, name='EGKL')
    runner.run()
