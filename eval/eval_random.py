from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
from utils.log import init_log
import time
import numpy as np 
from algos.policy.random import Random
import os
from env_wrapper.popenv import POPEnv
from utils.util import read_config
import yaml
from easydict import EasyDict as edict
from utils.util import args_type
from env_wrapper.draw.animate import gif_plot
from tabulate import tabulate
import json
import shutil
from utils.util import set_seed

mean_metrix_key = ['mean_tracking_ratio', 'mean_traking_fairness', 
              'mean_certainty_ratio', 'mean_collision_ratio',]
std_metrix_key = ['std_tracking_ratio', 'std_traking_fairness', 
              'std_certainty_ratio', 'std_collision_ratio',]

metrix_key = mean_metrix_key + std_metrix_key

def set_all_config(common_config_path, args, save_path):
    common_config = read_config(common_config_path)
    all_config = {}
    all_config.update(common_config)
    all_config = edict(all_config)
    all_config['learning_stage'] = args.learning_stage
    all_config['use_matrix'] = args.use_matrix
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            if key in all_config.keys():
                value_type = args_type(all_config[key])
                all_config[key] = value_type(value)
    for keys in common_config.keys():
        common_config[keys] = all_config[keys]
    with open(os.path.join(save_path, common_config_path.split(os.path.normpath('/'))[-1]), 'w') as f:
        yaml.dump(common_config, f,  default_flow_style=False)
    return all_config

def test_random(args_tuple):

    """
        args: 命令行参数
        run_dir: 日志保存目录
        config_dir: 配置目录
        observer_num: 观测者数量
        target_num: 逃逸者数量
        env_range: 环境范围
        test_episode: 测试轮数
        draw: 是否绘图
        test_index: 测试序列编号
        exp_num: 实验编号
    """
    args, run_dir, config_dir, _, observer_num, target_num, env_range, test_episode, draw, test_index, exp_num = args_tuple

    run_dir = os.path.join(run_dir, f'run{test_index}')
    mean_metrix_key = ['mean_tracking_ratio', 'mean_traking_fairness', 
                'mean_certainty_ratio', 'mean_collision_ratio',]
    std_metrix_key = ['std_tracking_ratio', 'std_traking_fairness', 
                'std_certainty_ratio', 'std_collision_ratio',]

    metrix_key = ['average_tracking_ratio', 'standard_deviation', 
                'average_certainty_ratio', 'average_collision_ratio']

    mean_dict = {key:0 for key in mean_metrix_key}
    std_dict = {key:0 for key in std_metrix_key}
    eval_dict_list = {key:[] for key in metrix_key}

    common_config_path = os.path.join(config_dir, 'common_config.yml')
    logger, save_path, log_file = init_log(run_dir)
    shutil.copy(common_config_path, save_path)

    env_config.env_range = env_range
    env_config.min_evader_num =  target_num
    env_config.max_evader_num = target_num
    env_config.max_pursuer_num = observer_num
    env_config.min_pursuer_num = observer_num
    base_env = BaseEnv(env_config)
    all_config = set_all_config(common_config_path, args, save_path)
    all_config.decision_dt = 1
    all_config.matrix_computation = 'sample'
    env = POPEnv(env=base_env, config=all_config)

    env_config_save_pth = os.path.join(save_path, 'env_config.json')
    with open(env_config_save_pth, 'w')  as f:
        json.dump(env_config.__dict__, f, indent=2)
    set_seed(args.seed*10+test_index)
    episode_score = []
    best_average_track_ratio = -np.inf
    time_start = time.time()
    policys = []
    for i in range(observer_num):
        policy = Random(all_config.use_matrix)
        policys.append(policy)
    env.set_render_flag()
    for episode in range(test_episode):
        episode_score.append(0)
        done = False
        env.reset()
        obs_dict = env.env_core.obs_dict
        step = 0
        while not done:

            actions = []
            for agent_idx, agent in enumerate(env.env_core.agents):
                if agent.group == 1:
                    continue
                action = policys[agent_idx].find_next_action(None, agent)
                actions.append(action)

            obs_dict, done, info  = env.env_core.step(actions)
            step += 1
            # if env.env_core.env_step % 10 == 0:
            #     env.render(env.fig)
            if env.env_core.env_step % 10 == 0:
                env._update_global_probability_matrix(obs_dict)
                env._get_local_matrix(obs_dict)
        total_time = round(time.time() - time_start, 2)
        metrix = info['metrix']
        if metrix.average_tracking_ratio > best_average_track_ratio:
            best_average_track_ratio = metrix.average_tracking_ratio
            if draw:
                env.save_info(save_path)
        for key in metrix_key:
            eval_dict_list[key].append(getattr(metrix, key))
        logger.info('phase: TEST, episodes: {}, episode_len: {}, tracking_ratio: {:.2f}, certainty_ratio: {:.2f}, standard_deviation: {:.2f}, best_tracking_ratio: {:.2f}, total_time: {:.2f}'.format(
                        episode, step, metrix.average_tracking_ratio, metrix.average_certainty_ratio, metrix.standard_deviation, best_average_track_ratio, total_time))
    if draw:
        gif_plot(info_file=save_path+f'/env_cfg.json', traj_file=save_path+f'/trajs.xlsx', 
                matrix_file=save_path+f'/matrix.npy', local_matrix_file=save_path+f'/local_agent_matrix.npy',plot_save_dir= save_path + os.path.normpath('/'), online=False, speedup_factor=10)

    mean_table = []
    std_table = []
    for i, key in enumerate(metrix_key):
        # mean 
        mean_dict[mean_metrix_key[i]] = np.mean(eval_dict_list[key])
        # standard_deviation
        std_dict[std_metrix_key[i]] = np.std(eval_dict_list[key])
        mean_table.append(mean_dict[mean_metrix_key[i]])
        std_table.append(std_dict[std_metrix_key[i]])
    print(tabulate([mean_table], headers=mean_metrix_key, tablefmt='fancy_grid'))
    print(tabulate([std_table], headers=std_metrix_key, tablefmt='fancy_grid'))
    total_dict = {'Observer_num': observer_num, 'Evader_Num': target_num, 'Exp_Num': exp_num,
                  'Env_Range': env_range[0][1], **mean_dict, **std_dict}
    return total_dict

 
