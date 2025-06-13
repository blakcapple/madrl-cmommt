"""
Written by Ruan Yudi 
"""
import os
import torch as T
from multiprocessing import Pool
from utils.arguments import read_agrs
from .eval_acmommt import test_acmommt
from .eval_pamts import test_pamts
from .eval_rl import test_rl
from .eval_random import test_random

mean_metrix_key = ['mean_tracking_ratio', 'mean_traking_fairness', 
              'mean_certainty_ratio', 'mean_collision_ratio',]
std_metrix_key = ['std_tracking_ratio', 'std_traking_fairness', 
              'std_certainty_ratio', 'std_collision_ratio',]

metrix_key = mean_metrix_key + std_metrix_key

def test_with_single_params(args, run_dir, config_dir):

    params_list = []
    total_test_episode = 10
    load_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'load_model')
    process_num = 1
    for i in range(process_num): 
        test_episode = int(total_test_episode / process_num)
        target_num = 8
        observer_num = 8
        env_max = 20
        env_range = [[0, env_max], [0, env_max]]
        exp_num = 0
        params_list.append((args, run_dir, config_dir, load_dir, observer_num, target_num, env_range, test_episode, True, i, exp_num))
    # 创建进程池并并行执行test_process
    with Pool(processes=process_num) as pool:
        if args.algo == 'random':
            test_fn = test_random
        elif args.algo == 'acmommt':
            test_fn = test_acmommt
        elif args.algo == 'pamts':
            test_fn = test_pamts
        else:
            test_fn = test_rl

        results = pool.map(test_fn, params_list)
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


def test_with_various_params(args, run_dir, config_dir):

    env_config_list = [['10', '2', '1', '1'], ['10', '2', '2', '2'], ['10', '2', '4', '3'],
                        ['10', '4', '2', '4'], ['10', '4', '4', '5'], ['10', '4', '8', '6'],
                        ['20', '8', '4', '7'], ['20', '8', '8', '8'], ['20', '8', '16', '9'],
                        ['20', '16', '8', '10'], ['20', '16', '16', '11'], ['20', '16', '32', '12'],
                        ['30', '18', '9', '13'], ['30', '18', '18', '14'], ['30', '18', '36', '15'],
                        ['30', '18', '9', '13'], ['30', '18', '18', '14'], ['30', '18', '36', '15'],
                        ['30', '18', '9', '13'], ['30', '18', '18', '14'], ['30', '18', '36', '15'],
                        ['30', '18', '9', '13'], ['30', '18', '18', '14'], ['30', '18', '36', '15'],
                        ['30', '36', '18', '16'], ['30', '36', '18', '16'], ['30', '36', '18', '16'], 
                        ['30', '36', '18', '16'], ['30', '36', '36', '17'], ['30', '36', '36', '17'],
                        ['30', '36', '36', '17'], ['30', '36', '36', '17'], ['30', '36', '36', '17'],
                        ['30', '36', '72', '18'], ['30', '36', '72', '18'], ['30', '36', '72', '18'],
                        ['30', '36', '72', '18'], ['30', '36', '72', '18'], ['30', '36', '72', '18'],]
    tmp_dict = {}
    for env_config in env_config_list:
        exp_num = int(env_config[-1])
        if exp_num not in tmp_dict:
            tmp_dict[exp_num] = 1
        else:
            tmp_dict[exp_num] += 1
    exp_test_episode = 100
    params_list = []
    load_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'load_model')
    process_num = 15
    for i in range(len(env_config_list)):
        test_episode = int(exp_test_episode / tmp_dict[int(env_config_list[i][-1])])
        env_config = env_config_list[i]
        env_max = int(env_config[0])
        target_num = int(env_config[2])
        observer_num = int(env_config[1])
        env_max = int(env_config[0])
        env_range = [[0, env_max], [0, env_max]]
        exp_num = int(env_config[-1])
        params_list.append((args, run_dir, config_dir, load_dir, observer_num, target_num, env_range, test_episode, False, i, exp_num))

    # 创建进程池并并行执行test_process
    with Pool(processes=process_num) as pool:
        if args.algo == 'random':
            test_fn = test_random
        elif args.algo == 'acmommt':
            test_fn = test_acmommt
        elif args.algo == 'pamts':
            test_fn = test_pamts
        else:
            test_fn = test_rl
        results = pool.map(test_fn, params_list)
        with open(str(run_dir)+'/total_res.csv', 'a') as f:
            import csv 
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            # 合并字典序列中实验序号相同的结果
            exp_num_list = {}
            new_results = [] # 合并后的结果
            for i, r in enumerate(results):
                if r['Exp_Num'] not in exp_num_list.keys():
                    exp_num_list[r['Exp_Num']] = [i]
                else:
                    exp_num_list[r['Exp_Num']].append(i)
            for key, value in exp_num_list.items():
                if len(value) > 1:
                    mean_res = {}
                    for i in value:
                        for k, v in results[i].items():
                            if k not in metrix_key:
                                mean_res[k] = v
                            else:
                                if k not in mean_res:
                                    mean_res[k] = v
                                else:
                                    mean_res[k] += v
                    for k in mean_res.keys(): 
                        if k in metrix_key:
                            mean_res[k] /= len(value)
                    writer.writerow(mean_res)
                    new_results.append(mean_res)
                else:
                    writer.writerow(results[value[0]]) 
                    new_results.append(results[value[0]])
        with open(str(run_dir)+'/mean_env_res.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=metrix_key+['Env_Range'], extrasaction='ignore')
            writer.writeheader()
            # 合并字典序列中环境范围相同的结果
            env_num_list = {}
            for i, r in enumerate(new_results):
                if r['Env_Range'] not in env_num_list.keys():
                    env_num_list[r['Env_Range']] = [i]
                else:
                    env_num_list[r['Env_Range']].append(i)
            for key, value in env_num_list.items():
                if len(value) > 1:
                    mean_res = {}
                    for i in value:
                        for k, v in new_results[i].items():
                            if k not in metrix_key:
                                mean_res[k] = v
                            else:
                                if k not in mean_res:
                                    mean_res[k] = v
                                else:
                                    mean_res[k] += v
                    for k in mean_res.keys():
                        if k in metrix_key:
                            mean_res[k] /= len(value)
                    writer.writerow(mean_res)
                else:
                    writer.writerow(new_results[i]) 
        with open(str(run_dir)+'/mean_res.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=metrix_key)
            writer.writeheader()
            # 计算平均指标  
            mean_res = {}
            for i in range(len(new_results)):
                for key, value in new_results[i].items():
                    if key in metrix_key:
                        if key not in mean_res:
                            mean_res[key] = value
                        else:
                            mean_res[key] += value
            for key in mean_res.keys():
                mean_res[key] /= len(new_results)
            writer.writerow(mean_res)

if __name__ == '__main__':
    args = read_agrs()
    
    T.set_num_threads(1) # 防止torch多线程导致的性能下降
    if args.learning_stage == 2:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/meta_policy')
    else:
        config_dir =  os.path.join(os.path.dirname(__file__), 'configs/default')
    from pathlib import Path
    run_dir = str(Path(os.path.dirname(os.path.abspath(__file__))) / 'data' / 'full_test')
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = os.path.join(run_dir, curr_run)
    test_with_various_params(args, run_dir, config_dir)