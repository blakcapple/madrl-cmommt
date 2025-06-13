import os
import shutil
from network.value.qfun import Qfun
from utils.log import init_log, log_info
from distributed_marl.dqn_apex.dqn_worker import DQNWorker
from distributed_marl.dqn_apex.dqn_learner import DQNLearner
from distributed_marl.dqn_apex.apex import ApeX
from .share import set_all_config, set_env

def train_apex(args, run_dir, config_dir):
    
    os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, 'dqn_apex.yml')

    # parameter setting
    logger, save_path, log_file = init_log(run_dir)
    all_config = set_all_config(common_config_path, train_config_path, args)
    all_config.save_path = save_path
    
    env = set_env(all_config, args.seed)
    env_info = env.get_info()
    log_info(logger, all_config, env_info, args)
    obs_space = env.observation_space
    del env
    
    # copy config 
    shutil.copy(train_config_path, save_path)
    shutil.copy(common_config_path, save_path)

    brain = Qfun(all_config, env_info)
    
    if args.load:
        mac_path = os.path.join(args.load_dir, "mac.pth")
        brain.load(mac_path)
    
    ApeXDQN = ApeX(DQNWorker, DQNLearner, brain, env_info, all_config, obs_space, set_env)
    ApeXDQN.spawn()
    ApeXDQN.train()