"""
Written by Ruan Yudi 
"""
import os
import torch as T
import wandb
from utils.log import init_log
from utils.util import set_seed
from algos.runner.dqn import Trainer, RTrainer
from algos.agents import DQNAgent
from utils.log import log_info
from utils.util import set_all_config, set_env, set_policy
from satenv.configs.base_cfg import env_config
from utils.util import set_parallel_env

def train_dqn(args, run_dir, config_dir):
    if args.learning_stage == 2:
        T.set_num_threads(1) 
    common_config_path = os.path.join(config_dir, 'common_config.yml')        
    if args.use_rnn:
        train_config_path = os.path.join(config_dir, f'{args.algo}_rnn.yml')
    else:
        train_config_path = os.path.join(config_dir, f'{args.algo}.yml')
    device = T.device('cuda:0' if T.cuda.is_available() and not args.use_cpu else 'cpu')
    # set config
    logger, save_path, log_file = init_log(run_dir)
    all_config = set_all_config(common_config_path, train_config_path, args, save_path)
    all_config.save_path = save_path
    all_config.device = device
    
    # init_wanb
    if args.use_wandb:
        wandb.init(project=all_config.wandb_project, entity=all_config.wandb_entity, group=all_config.wandb_group)
    eval_envs = set_parallel_env(all_config, env_config, env_num=all_config.n_eval_rollout_threads)
    env = set_env(all_config, args.seed, env_config)
        
    env_info = env.get_info()
    
    set_seed(args.seed)
    log_info(logger, all_config, env_info, args)
    
    logger.info('device: {}'.format(device))

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
    trainer.eval_envs = eval_envs
    # main loop
    trainer.warmup()
    trainer.train()






