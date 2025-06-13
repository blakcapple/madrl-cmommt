import os
import shutil
from utils.log import init_log
from utils.util import set_seed
from algos.runner.dqn.evaluator import Evaluator
from satenv.configs.base_cfg import env_config
from utils.util import set_all_config, set_env, set_policy
from algos.agents import DQNAgent, PPOAgent
from utils.log import log_test_info

mean_metrix_key = ['mean_tracking_ratio', 'mean_traking_fairness', 
              'mean_certainty_ratio', 'mean_collision_ratio',]
std_metrix_key = ['std_tracking_ratio', 'std_traking_fairness', 
              'std_certainty_ratio', 'std_collision_ratio',]

metrix_key = mean_metrix_key + std_metrix_key

def test_rl(args_tuple):

    """
    args: 命令行参数
    run_dir: 日志保存目录
    config_dir: 配置目录
    load_dir: 模型加载目录
    observer_num: 观察者数量
    target_num: 目标数量
    env_range: 环境范围
    test_episode: 测试轮数
    draw: 是否绘图
    test_index: 测试序列编号
    exp_num: 实验序号
    """
    args, run_dir, config_dir, load_dir, observer_num, target_num, env_range, test_episode, draw, test_index, exp_num = args_tuple
    common_config_path = os.path.join(config_dir, 'common_config.yml')
    train_config_path = os.path.join(config_dir, f'{args.algo}.yml')

    # 创建子目录
    curr_run = f'run{test_index}'
    run_dir = os.path.join(run_dir, curr_run)
    # parameter setting
    logger, save_path, log_file = init_log(run_dir, test_index)
    env_config.test = True 
    device = 'cpu'
    all_config = set_all_config(common_config_path, train_config_path, args, save_path)
    all_config.save_path = save_path
    all_config.device = device
    # set env
    # env_config.total_steps = 500
    env_config.max_evader_num = target_num
    env_config.min_evader_num = target_num
    env_config.min_pursuer_num = observer_num
    env_config.max_pursuer_num = observer_num
    env_config.env_range = env_range
    env = set_env(all_config, args.seed, env_config)
    env_info = env.get_info()
    set_seed(args.seed*10+test_index)
    log_test_info(logger, all_config, env_info)
    # device setting
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
    if 'ppo' in args.algo:
        from network.ac.ac import PPOAC
        policy = PPOAC(all_config, env_info, device)
    else:
        policy = set_policy(args.algo, all_config, env_info, device)

    # build agents  
    mac_path = os.path.join(load_dir, f"mac_{args.learning_stage}.pth")
    shutil.copy(mac_path, save_path)
    if 'iql' in args.algo or args.algo == 'vdn' or args.algo == 'qmix' or 'apex' in args.algo:
        agent = DQNAgent(env_info, all_config, device, policy)
    elif 'ppo' in args.algo:
        agent = PPOAgent(policy, device)
    agent.load_weight(mac_path)
    logger.info(f'test index: {test_index}')
    evaluator = Evaluator(env, env_info, agent, save_file, args.use_wandb, logger, log_file, all_config)
    
    record_dict, mean_dict, std_dict = evaluator.test(test_episode, save_path, online=args.show_online, draw=draw)
    total_dict = {'Observer_num': observer_num, 'Evader_Num': target_num, 'Exp_Num': exp_num,
                  'Env_Range': env_range[0][1], **record_dict, **mean_dict, **std_dict}

    return total_dict