"""
Written by Ruan Yudi 
"""
import os
import torch as T
import shutil
from utils.log import init_log
from utils.util import set_seed
from algos.runner.dqn.evaluator import Evaluator
from satenv.configs.base_cfg import env_config
from pathlib import Path
from .share import set_all_config, set_env, set_policy
from algos.agents import DQNAgent, PPOAgent

def full_test(args, config_dir):

    load_dir = Path(__file__).parent.parent / 'load_model'
    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, f'{args.algo}.yml')

    # parameter setting
    logger, save_path, log_file = init_log(args.output_dir)
    env_config.test = True
    all_config = set_all_config(common_config_path, train_config_path, args)
    
    evader_num_list = [4, 8, 16]
    env_range_list = [10, 20, 30]
    eval_all_dict_list = [] # 记录所有的评估字典
    for i in range(9):
        # set env
        evader_num = evader_num_list[int(i % 3)]
        env_max = env_range_list[int(i//3)]
        env_range = [[0, env_max], [0, env_max]]
        env_config.evader_num = evader_num
        env_config.ENV_RANGE = env_range
        env_config.pursuer_num = 8
        env = set_env(all_config, args, env_config)
        env_info = env.get_info()
        set_seed(args.seed)

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

        # build agents  
        policy = set_policy(args.algo, all_config, env_info, device)
        # build agents  
        mac_path = os.path.join(load_dir, f"mac_{args.learning_stage}.pth")
        if args.algo == 'iql' or args.algo == 'vdn' or args.algo == 'qmix':
            agent = DQNAgent(env_info, all_config, device, policy)
        elif args.algo == 'ippo':
            agent = PPOAgent(policy, device)

        evaluator = Evaluator(env, env_info, agent, save_file, args.use_wandb, logger, log_file, all_config)
        # main loop
        stage = args.learning_stage
        mac_path = os.path.join(load_dir, f"mac_{stage}.pth")

        agent.load_weight(mac_path, None)
        record_dict, mean_dict, std_dict = evaluator.test(250, save_path, show=args.show, index=i, draw=True)
        total_dict = {**record_dict, **mean_dict, **std_dict}
        eval_all_dict_list.append(total_dict)
    
    with open(save_path+'/record.csv', 'a') as f:
        import csv 
        writer = csv.DictWriter(f, fieldnames=total_dict.keys())
        writer.writeheader()
        for i in range(len(eval_all_dict_list)):
            writer.writerow(eval_all_dict_list[i])




