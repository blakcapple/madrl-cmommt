import os
import torch as T
import shutil
from utils.log import init_log, log_info
from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
from env_wrapper.popenv import POPEnv
from env_wrapper.vecenv.env_wrappers import ShareSubprocVecEnv
from algos.runner.ppo.ppo_runner import PPORunner
from .share import set_env, set_all_config
import wandb

def train_mappo(args, run_dir, config_dir):
    
    def make_env(all_config):
        env_num = all_config.n_rollout_threads
        def get_env_fn(rank):
            def init_env():
                if all_config.learning_stage == 2:
                    from env_wrapper.meta_env import MetaEnv
                    from network.value.qfun import Qfun
                    env_info = POPEnv(BaseEnv(env_config), all_config).get_info()
                    track_policy = Qfun(all_config, env_info)
                    search_policy =  Qfun(all_config, env_info)
                    env = MetaEnv(BaseEnv(env_config), all_config)
                    env.set_model(track_policy, search_policy)
                else:
                    env = POPEnv(BaseEnv(env_config), all_config)
                env.env_core.seed(rank)
                return env
            return init_env
        if env_num == 1:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(env_num)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(env_num)])
        
    os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.    
    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, f'{args.algo}.yml')
    device = T.device('cuda:0' if T.cuda.is_available() and not args.use_cpu else 'cpu')
    # parameter setting
    logger, save_path, log_file = init_log(run_dir)
    # copy config 
    shutil.copy(train_config_path, save_path)
    shutil.copy(common_config_path, save_path)
    all_config = set_all_config(common_config_path, train_config_path, args)
    all_config.run_dir = run_dir
    all_config.save_path = save_path
    all_config.device = device
    # set env
    train_env = make_env(all_config)
    eval_env = set_env(all_config, args.seed, env_config)
    env_info = eval_env.get_info()
    all_config.env_info = env_info
    # log_info(logger, all_config, env_info, args)
    
    from network.ac.ac import PPOAC
    policy = PPOAC(all_config, env_info, device)
    runner = PPORunner(all_config, train_env, policy, eval_env=eval_env)

    # init_wanb
    if args.use_wandb:
        wandb.init(project="ShadowClone", entity="the-one", group='CMOMMT4-1', config=all_config, name='EGKL')
    runner.run()
