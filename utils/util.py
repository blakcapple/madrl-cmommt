from typing import Dict, Tuple, Union
import numpy as np
from gym import spaces
from env_wrapper.popenv import POPEnv
from satenv.base import BaseEnv
from easydict import EasyDict as edict
import os 
import json
from pathlib import Path
import torch
import shutil
import yaml
from env_wrapper.vecenv.env_wrappers import ShareSubprocVecEnv
import yaml
from yaml import Loader
import torch as T 
import numpy as np 

def read_config(config_path: str):
    with open(config_path, "rb") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)

    return cfg

def set_seed(seed):
    # set rseed
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed) 

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")

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
        track_policy.load_state_dict(torch.load(track_pth, map_location=all_config.device), strict=False)
        search_policy.load_state_dict(torch.load(search_pth, map_location=all_config.device), strict=False)
        shutil.copy(track_pth, all_config.save_path)
        shutil.copy(search_pth, all_config.save_path)
        env = MetaEnv(BaseEnv(env_config), all_config)
        env.set_model(track_policy, search_policy)
    else:
        env = POPEnv(BaseEnv(env_config), all_config)
     
    return env

def set_parallel_env(all_config, env_config, env_num):
    def get_env_fn(rank):
        def init_env():
            if all_config.learning_stage == 2:
                from env_wrapper.meta_env import MetaEnv
                from network.value.qfun import Qfun
                env_info = POPEnv(BaseEnv(env_config), all_config).get_info()
                track_policy = Qfun(all_config, env_info)
                search_policy =  Qfun(all_config, env_info)
                track_pth = Path(__file__).parent.parent / 'load_model' / 'mac_1.pth'
                search_pth = Path(__file__).parent.parent / 'load_model' / 'mac_0.pth'
                track_policy.load_state_dict(torch.load(track_pth, map_location='cpu'), strict=False)
                search_policy.load_state_dict(torch.load(search_pth, map_location='cpu'), strict=False)
                env = MetaEnv(BaseEnv(env_config), all_config)
                env.set_model(track_policy, search_policy)
            else:
                env = POPEnv(BaseEnv(env_config), all_config)
            env.env_core.set_seed(rank)
            return env
        return init_env
    
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(env_num)])


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def set_all_config(common_config_path, train_config_path, args, save_path):
    
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
            if key in all_config.keys():
                value_type = args_type(all_config[key])
                all_config[key] = value_type(value)
    for keys in train_config.keys():
        train_config[keys] = all_config[keys]
    for keys in common_config.keys():
        common_config[keys] = all_config[keys]
    with open(os.path.join(save_path, train_config_path.split(os.path.normpath('/'))[-1]), 'w') as f:
        yaml.dump(train_config, f,  default_flow_style=False)
    with open(os.path.join(save_path, common_config_path.split(os.path.normpath('/'))[-1]), 'w') as f:
        yaml.dump(common_config, f,  default_flow_style=False)
    with open(os.path.join(save_path, 'all_config.yml'), 'w') as f:
        yaml.dump(all_config, f,  default_flow_style=False)
    return all_config

def set_policy(algo, all_config, env_info, device): 
    
    from network.value.qfun import Qfun
    from network.value.mixer import QMixer
    from network.value.vdn import VDNMixer
    from algos.policy.iql import IQL
    from algos.policy.qmix import QMix
    Q = Qfun(all_config, env_info) 
    if 'iql' in algo or 'apex' in algo:     
        policy = IQL(env_info, all_config, device, Q)
    elif algo == 'qmix':
        mixer = QMixer(all_config, env_info)
        policy = QMix(env_info, all_config, device, Q, mixer)
    elif algo == 'vdn':
        mixer = VDNMixer()
        policy = QMix(env_info, all_config, device, Q, mixer)
    else:
        raise NotImplementedError

    return policy