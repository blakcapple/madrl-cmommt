"""
written by Ruan Yudi 
2022.6.28
"""
import numpy as np
import wandb
from utils.plot import plot_rw
import sys 
sys.path.append('./maca/draw')
from env_wrapper.draw.animate import gif_plot
from tabulate import tabulate
import time 

mean_metrix_key = ['mean_tracking_ratio', 'mean_traking_fairness', 
              'mean_certainty_ratio', 'mean_collision_ratio',]
std_metrix_key = ['std_tracking_ratio', 'std_traking_fairness', 
              'std_certainty_ratio', 'std_collision_ratio',]

metrix_key = ['average_tracking_ratio', 'standard_deviation', 
              'average_certainty_ratio', 'average_collision_ratio']

class Evaluator(object):
    '''Train process'''
    def __init__(self, env, env_info, agent, save_file, use_wandb, logger, log_file, train_config):
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.save_file = save_file
        self.use_wandb = use_wandb
        self.logger = logger
        self.log_file = log_file
        self.cfg = train_config
        self.train_step = 0
        
        self.mean_dict = {key:0 for key in mean_metrix_key}
        self.std_dict = {key:0 for key in std_metrix_key}
        self.eval_dict_list = {key:[] for key in metrix_key}
        # self.record_dict = {'Evader_Speed': env_info.evader_speed, 
        #                     'Evader_Num': env_info.evader_num,
        #                     'Pursuer_Num': env_info.pursuer_num,
        #                     'Env_Range': env_info.env_range[0][1],
        #                       }
        self.record_dict = {}
        
    def test(self, episode_num, save_path, online, draw=False, index=0):
        
        self.env.set_render_flag()
        """
        智能体测试回合
        """
        episode_score = []
        best_score = -np.inf
        time_start = time.time()
        for episode in range(episode_num):
            episode_score.append(0)
            done = False
            obs = self.env.reset()
            step = 0
            while not done:
                # 选择动作（此时的动作未乘上单步时长）
                actions = self.agent.select_actions(obs, deterministic=True)
                # 执行动作 （此时的动作乘上单步时长）
                obs_new, reward, done, info, *_ = self.env.step(actions)
                step +=1
                # 更新状态
                obs = obs_new
                episode_score[-1] += np.mean(reward).item()

            if best_score < episode_score[-1]:
                best_score = episode_score[-1]
                import os 
                if not os.path.exists(save_path+f'/log_{index}'):
                    os.mkdir(save_path+f'/log_{index}')
                self.env.save_info(save_path+f'/log_{index}')
            for key in metrix_key:
                self.eval_dict_list[key].append(getattr(info, key))
            total_time = round(time.time() - time_start, 2)
            self.logger.info('phase: TEST, episodes: {}, episode_len: {}, tracking_ratio: {:.2f}, certainty_ratio: {:.2f}, '
                             'standard_deviation: {:.2f}, total_time: {:.2f}, observer_num: {}, targeta_num: {}'.format(
                             episode, step, info.average_tracking_ratio, info.average_certainty_ratio, info.standard_deviation,
                             total_time, self.env.pursuer_num, self.env.evader_num))
        mean_table = []
        std_table = []
        for i, key in enumerate(metrix_key):
            # mean 
            self.mean_dict[mean_metrix_key[i]] = np.mean(self.eval_dict_list[key])
            # standard_deviation
            self.std_dict[std_metrix_key[i]] = np.std(self.eval_dict_list[key])
            mean_table.append(self.mean_dict[mean_metrix_key[i]])
            std_table.append(self.std_dict[std_metrix_key[i]])
        print(tabulate([mean_table], headers=mean_metrix_key, tablefmt='fancy_grid'))
        print(tabulate([std_table], headers=std_metrix_key, tablefmt='fancy_grid'))

        self.logger.info('EVALUATION DONE')
        if draw:
            gif_plot(info_file=save_path+f'/log_{index}/env_cfg.json', traj_file=save_path+f'/log_{index}/trajs.xlsx', 
                    matrix_file=save_path+f'/log_{index}/matrix.npy', local_matrix_file=save_path+f'/log_{index}/local_agent_matrix.npy',plot_save_dir= save_path + os.path.normpath('/'), online=online, speedup_factor=10)
        plot_rw(self.log_file, window_size=10, phase='test')

        return self.record_dict, self.mean_dict, self.std_dict

    def log_train(self, train_info, train_step):
        for k, v in train_info.items():
            wandb.log({k:v}, step=train_step)