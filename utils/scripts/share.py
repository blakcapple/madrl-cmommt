from env_wrapper.popenv import POPEnv
from satenv.base import BaseEnv
from utils.util import read_config
from easydict import EasyDict as edict
import os 
import json
from pathlib import Path
import torch
import shutil

def set_env(all_config, seed, env_config=None):
    if env_config is None:
        from satenv.configs.base_cfg import env_config
    # 将环境信息保存到文件中
    env_config_save_pth = os.path.join(all_config.save_path, 'env_config.json')
    with open(env_config_save_pth, 'w')  as f:
        json.dump(env_config.__dict__, f, indent=2)
    if all_config.learning_stage == 2:
        from env_wrapper.meta_env import MetaEnv
        from network.value.qfun import Qfun
        env_info = POPEnv(BaseEnv(env_config), all_config).get_info()
        track_policy = Qfun(all_config, env_info)
        search_policy = Qfun(all_config, env_info)
        track_pth = Path(__file__).parent.parent / 'load_model' / 'mac_1.pth'
        search_pth = Path(__file__).parent.parent / 'load_model' / 'mac_0.pth'
        track_policy.load_state_dict(torch.load(track_pth))
        search_policy.load_state_dict(torch.load(search_pth))
        shutil.copy(track_pth, all_config.save_path)
        shutil.copy(search_pth, all_config.save_path)
        env = MetaEnv(BaseEnv(env_config), all_config)
        env.set_model(track_policy, search_policy)
    else:
        env = POPEnv(BaseEnv(env_config), all_config)
    env.env_core.seed(seed)
     
    return env

def set_all_config(common_config_path, train_config_path, args):
    
    train_config = read_config(train_config_path)
    common_config = read_config(common_config_path)
    all_config = {}
    all_config.update(common_config)
    all_config.update(train_config)
    # 阶段1训练时，每一个观测中只包含一个逃脱者
    if args.learning_stage == 1:
        all_config['max_evader_in_obs'] = 1
    all_config['pursuer_policy'] = args.algo
    all_config['use_matrix'] = args.use_matrix
    all_config['learning_stage'] = args.learning_stage
    all_config['use_rnn'] = args.use_rnn
    all_config['use_wandb'] = args.use_wandb
    all_config = edict(all_config)
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            all_config[key] = value
    
    return all_config

def set_policy(algo, all_config, env_info, device): 
    
    from network.value.qfun import Qfun
    from network.value.mixer import QMixer
    from network.value.vdn import VDNMixer
    from algos.policy.iql import IQL
    from algos.policy.qmix import QMix
    from network.ac.ac import PPOAC
    Q = Qfun(all_config, env_info) 
    if 'iql' in algo:
        policy = IQL(env_info, all_config, device, Q)
    elif algo == 'qmix':
        mixer = QMixer(all_config, env_info)
        policy = QMix(env_info, all_config, device, Q, mixer)
    elif algo == 'vdn':
        mixer = VDNMixer()
        policy = QMix(env_info, all_config, device, Q, mixer)
    elif algo == 'ippo':
        policy = PPOAC(all_config, env_info, device)
    else:
        raise NotImplementedError

    return policy
    