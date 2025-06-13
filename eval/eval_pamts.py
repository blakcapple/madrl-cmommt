from satenv.base import BaseEnv
from satenv.configs.base_cfg import env_config
from utils.log import init_log
import time
import numpy as np 
from algos.policy.pamts import PAMTS
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
from multiprocessing import Pool
from satenv.env_utils.utils import l2norm
from copy import deepcopy
from utils.util import set_seed

def merge_observe_agent(obs_dict, agents, pursuer_num):
    who_observe_evader_dict = obs_dict['who_observe_evader_dict']
    observe_target = {i:[] for i in range(pursuer_num)}
    for evader_id in who_observe_evader_dict.keys():
        if len(who_observe_evader_dict[evader_id]) > 0:
            pursuer_list = who_observe_evader_dict[evader_id]
            distances = [l2norm(agents[pursuer_id].pos, agents[evader_id].pos) for pursuer_id in pursuer_list]
            closest_index = np.argmin(distances)
            closest_pursuer_id = pursuer_list[closest_index]
            observe_target[closest_pursuer_id].append(evader_id)
    
    return observe_target

def merge_observe_matrix(obs_dict, observe_matrix):
    # merge
    merge_before_matrix = deepcopy(observe_matrix)
    for index in range(len(observe_matrix)):
        friend_list = obs_dict['friend_observe_dict'][index]
        for agent_id in friend_list:
            observe_matrix[index] = np.maximum(observe_matrix[index], observe_matrix[agent_id])
    return observe_matrix

def set_all_config(common_config_path, args, save_path):

    common_config = read_config(common_config_path)
    all_config = {}
    all_config.update(common_config)
    all_config = edict(all_config)
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            if key in all_config.keys():
                value_type = args_type(all_config[key])
                all_config[key] = value_type(value)
    all_config['learning_stage'] = args.learning_stage
    all_config['use_matrix'] = args.use_matrix
    for keys in common_config.keys():
        common_config[keys] = all_config[keys]
    with open(os.path.join(save_path, common_config_path.split(os.path.normpath('/'))[-1]), 'w') as f:
        yaml.dump(common_config, f,  default_flow_style=False)
    return all_config

def test_pamts(args_tuple):

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
    env = POPEnv(env=base_env, config=all_config)
    set_seed(test_index)
    env_config_save_pth = os.path.join(save_path, 'env_config.json')
    with open(env_config_save_pth, 'w')  as f:
        json.dump(env_config.__dict__, f, indent=2)

    episode_score = []
    best_average_track_ratio = -np.inf
    time_start = time.time()
    env.set_render_flag()
    for episode in range(test_episode):
        episode_score.append(0)
        done = False
        env.reset()
        obs_dict = env.env_core.obs_dict
        observe_target = merge_observe_agent(obs_dict, env.env_core.agents, env.pursuer_num)
        friend_observe_dict = obs_dict['friend_observe_dict']
        agents = []
        observe_matrix = []
        for pursuer in env.env_core.pursuers:
            agent = PAMTS(pursuer.id, env_config.search_radius, env_config.env_range[0][1], env.evader_num, env.pursuer_num)
            agents.append(agent)
            observe_matrix.append(agents[pursuer.id].observe_matrix)
            agent.agent_pos = pursuer.pos
            agent.heading = pursuer.heading
        env_step = 0
        while not done:
            actions = []
            for pursuer in env.env_core.pursuers:
                agent_id = pursuer.id
                info ={'observe_friend': friend_observe_dict[agent_id], 
                       'observe_target': observe_target[agent_id], 
                       'observe_matrix': observe_matrix[agent_id],
                       'friend_find_target_num': len(obs_dict['pursuer_observe_evader_dict'][agent_id]),
                       'observe_target_pos': [env.env_core.agents[evader_id].pos for evader_id in observe_target[agent_id]],
                       'env_step': env_step}
                action = agents[agent_id].find_next_action(info)
                actions.append(action)
            for i in range(5):
                obs_dict, done, info  = env.env_core.step(actions)
            env_step += 1
            for agent in agents:
                agent.agent_pos = env.env_core.agents[agent.agent_id].pos
                agent.heading = env.env_core.agents[agent.agent_id].heading
            if env.env_core.env_step % 10 == 0:
                env.render(env.fig)
                # env.render2(env.fig, matrix=observe_matrix[0]/10, matrix_grid=0.5)
            for agent in agents:
                agent._update_observe_matrix(env_step)
            observe_matrix = merge_observe_matrix(obs_dict, [agent.observe_matrix for agent in agents])
            observe_target = merge_observe_agent(obs_dict, env.env_core.agents, env.pursuer_num)
        metrix = info['metrix']
        total_time = round(time.time() - time_start, 2)
        if metrix.average_tracking_ratio > best_average_track_ratio:
            best_average_track_ratio = metrix.average_tracking_ratio
            if draw:
                env.save_info(save_path)
        for key in metrix_key:
            eval_dict_list[key].append(getattr(metrix, key))
        logger.info('phase: TEST, episodes: {}, episode_len: {}, tracking_ratio: {:.2f}, certainty_ratio: {:.2f}, standard_deviation: {:.2f}, best_tracking_ratio: {:.2f}, total_time: {:.2f}'.format(
                        episode, env_step, metrix.average_tracking_ratio, metrix.average_certainty_ratio, metrix.standard_deviation, best_average_track_ratio, total_time))
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
        
def test_with_various_params(args, run_dir, config_dir):

    params_list = []
    evader_num_list = [4, 8, 16]
    env_range_list = [10, 20, 30]
    test_episode = 250
    process_num = 9
    for i in range(9): 
        evader_num = evader_num_list[int(i % 3)]
        env_max = env_range_list[int(i//3)]
        env_range = [[0, env_max], [0, env_max]]
        params_list.append((args, run_dir, config_dir, evader_num, env_range, test_episode, False, i))
    # test_pamts((args, run_dir, config_dir, 8, [[0, 20], [0, 20]], 1, True, 0))
    # 创建进程池并并行执行test_process
    with Pool(processes=process_num) as pool:
        results = pool.map(test_pamts, params_list)
        with open(str(run_dir)+'/record.csv', 'a') as f:
            import csv 
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for i in range(len(results)):
                writer.writerow(results[i])   
            mean_res = {}
            for i in range(len(results)):
                for key, value in results[i].items():
                    if key == 'Evader_Num' or key == 'Env_Range':
                        continue
                    if key not in mean_res:
                        mean_res[key] = value
                    else:
                        mean_res[key] += value
            for key in mean_res.keys():
                mean_res[key] /= len(results)
            writer.writerow(mean_res)

 
