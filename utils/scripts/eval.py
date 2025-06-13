"""
Written by Ruan Yudi 
"""
import os
from pathlib import Path
import torch as T
import shutil
from utils.log import init_log
from utils.util import set_seed
from algos.runner.dqn.evaluator import Evaluator
from algos.agents import DQNAgent, PPOAgent
from satenv.configs.base_cfg import env_config
from pathlib import Path
from utils.log import log_test_info
from .share import set_all_config, set_env, set_policy

def test(args, run_dir, config_dir):
    
    load_dir = Path(__file__).parent.parent / 'load_model'
    
    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, f'{args.algo}.yml')

    # parameter setting
    logger, save_path, log_file = init_log(run_dir)
    env_config.test = True
    # env_config.evader_speed = 0.1
    # env_config.max_evader_num = 64
    # env_config.min_evader_num = 64
    # env_config.min_pursuer_num = 16
    # env_config.max_pursuer_num = 16
    # env_config.comm_radius = 5
    all_config = set_all_config(common_config_path, train_config_path, args)
    all_config.save_path = save_path
    # set env
    env_config.evader_policy = 'random'
    env = set_env(all_config, args.seed, env_config)
    env_info = env.get_info()
    set_seed(args.seed)
    log_test_info(logger, all_config, env_info)
    # device setting
    device = T.device('cuda:0' if T.cuda.is_available() and not args.use_cpu else 'cpu')
    logger.info('device: {}'.format(device))

    # copy config 
    shutil.copy(train_config_path, save_path)
    shutil.copy(common_config_path, save_path)

    # model saved path
    if all_config.action_type == 'discrete':
        mac_model_path = os.path.join(save_path, "mac.pth")
        mixer_model_path = os.path.join(save_path, "mixer.pth")
        save_file = dict(mac_path=mac_model_path, 
                        mixer_path=mixer_model_path)
    else:
        raise NotImplementedError

    policy = set_policy(args.algo, all_config, env_info, device)
    # build agents  
    mac_path = os.path.join(load_dir, f"mac_{args.learning_stage}.pth")
    if args.algo == 'iql' or args.algo == 'vdn' or args.algo == 'qmix':
        agent = DQNAgent(env_info, all_config, device, policy)
    elif args.algo == 'ippo':
        agent = PPOAgent(policy, device)
    agent.load_weight(mac_path)
        
    evaluator = Evaluator(env, env_info, agent, save_file, args.use_wandb, logger, log_file, all_config)
    record_dict, mean_dict, std_dict = evaluator.test(1, save_path, online=args.show_online, draw=True)
    total_dict = {**record_dict, **mean_dict, **std_dict}

    with open(save_path+'/record.csv', 'a') as f:
        import csv 
        writer = csv.DictWriter(f, fieldnames=total_dict.keys())
        writer.writeheader()
        writer.writerow(total_dict)