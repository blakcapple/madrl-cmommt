"""
Written by Ruan Yudi 
"""
import os
import torch as T
import shutil
import wandb
from utils.log import init_log
from utils.util import set_seed
from algos.runner.dqn import Trainer, RTrainer
from algos.agents import DQNAgent
from utils.log import log_info
from .share import set_all_config, set_env, set_policy
from satenv.configs.base_cfg import env_config

def train_dqn(args, run_dir, config_dir):
    
    common_config_path = os.path.join(config_dir, 'common_config.yml')        
    if args.use_rnn:
        train_config_path = os.path.join(config_dir, f'{args.algo}_rnn.yml')
    else:
        train_config_path = os.path.join(config_dir, f'{args.algo}.yml')
    
    # set config
    logger, save_path, log_file = init_log(run_dir)
    all_config = set_all_config(common_config_path, train_config_path, args)
    all_config.save_path = save_path
    
    # init_wanb
    if args.use_wandb:
        wandb.init(project="Pursuit-Evasion", entity="the-one", group='Simple Thread')

    # set env
    env = set_env(all_config, args.seed, env_config)
        
    env_info = env.get_info()
    
    set_seed(args.seed)
    log_info(logger, all_config, env_info, args)
    
    # device setting
    device = T.device('cuda:0' if T.cuda.is_available() and not args.use_cpu else 'cpu')
    logger.info('device: {}'.format(device))

    # copy config 
    shutil.copy(train_config_path, save_path)
    shutil.copy(common_config_path, save_path)

    # model saved path
    if all_config.action_type == 'discrete':
        mac_model_path = os.path.join(save_path, f"mac_{args.learning_stage}.pth")
        mixer_model_path = os.path.join(save_path, f"mixer_{args.learning_stage}.pth")
        save_file = dict(mac_path=mac_model_path, 
                        mixer_path=mixer_model_path)
    else:
        raise NotImplementedError
    
    # set trainer
    policy = set_policy(args.algo, all_config, env_info, device)
    agent = DQNAgent(env_info, all_config, device, policy)
    if all_config['use_rnn']:
        trainer = RTrainer(env, env_info, agent, save_file, logger, log_file, all_config)
    else:
        trainer = Trainer(env, env_info, agent, save_file, logger, log_file, all_config)
        
    # main loop
    trainer.warmup()
    trainer.train()






