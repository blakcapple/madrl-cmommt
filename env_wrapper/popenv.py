"""
Partilly Observed Pursuit-Evasion Environment Based on base env

Written by Ruan Yudi 2022.5.25
"""
from gym.spaces import Dict, Box, Discrete
from satenv.base import BaseEnv
import numpy as np 
from easydict import EasyDict as edict
import pandas as pd 
import os 
import json
from .env_utils.matrix import update_probability_matrix, merge_local_probability_matrix, create_matrix
from .env_utils.utils import l2norm
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from env_wrapper.draw.geometry import get_2d_car_model, get_2d_uav_model
from env_wrapper.draw.vis_util import rgba2rgb
from matplotlib.patches import Circle


# 动作空间
class Actions:
    def __init__(self, max_action, action_num, type='discrete'):
        self.max_action = max_action
        if type == 'discrete':
            self.map = np.linspace(-self.max_action, self.max_action, action_num, endpoint=True)  # 动作：角速度 弧度/s
            self.num = len(self.map)  # 动作空间数目，共61个
        else:
            self.min = -self.max_action
            self.max = self.max_action

# 评估指标
EvalMetrix = namedtuple('metrix', 
                        ['average_tracking_ratio', 
                        'standard_deviation', 
                        'average_certainty_ratio',
                        'average_collision_ratio',
                        ])

class POPEnv:

    def __init__(self, env:BaseEnv, config):
        
        # get info from base env
        self.env_core = env
        self.env_info = env.get_info()
        self.agent_num = env._max_pursuer_num # 取环境中最大的追逐者数量作为智能体的数量（不足的则补齐）
        self.max_env_range = self.env_core.env_range[0][1] # 环境最大范围，这里没有考虑x方向和y方向的范围不同的情况
        self.actions = Actions(self.env_core.max_angular_v, config.action_num, config.action_type)        

        # config setting
        self.config = config 
        self.gamma = config.gamma # 环境概率矩阵衰变系数
        self.matrix_grid = config.grid # 环境概率矩阵单元格的大小
        self.local_matrix_grid = config.local_grid 
        self.local_matrix_shape = config.local_matrix_shape
        self.friend_map_shape = config.friend_map_shape
        self.target_map_shape = config.target_map_shape
        self.friend_map_grid = config.friend_map_grid
        self.target_map_grid = config.target_map_grid
        self._collision_avoidance = config.collision_avoidance # 是否考虑避障
        self._learning_stage = config.learning_stage # 学习阶段
        self._save_matrix = config.save_matrix # 是否保存所有的概率矩阵信息
        self._ignore_speed = config.ignore_speed # 在神经网络输入中忽略speed这一项（在训练中speed是固定的）
        self._use_map_obs = config.use_map_obs # 是否将obs构建成map的形式
        self._use_matrix = config.use_matrix # 是否利用概率矩阵
        self._decision_dt = config.decision_dt # 决策间隔时间
        self.use_global_info = config.use_global_info # 是否利用全局观测
        self.use_agent_specific_global_info = config.agent_specific_info # 全局信息是否是以智能体为中心的
        self.use_global_matrix = config.use_global_matrix # 是否利用全局的概率矩阵
        self._track_weight = config.track_weight # 奖励权重
        self._explore_weight = config.explore_weight
        self._collision_weight = config.collision_weight
        self._overlapping_punishment_weight = config.overlapping_punishment_weight
        self._repeat_track_punishment_weight = config.repeat_track_punishment_weight
        self._matrix_computation = config.matrix_computation
        self.matrix_update_interval = config.matrix_update_interval
        if config.action_type == 'discrete':
            self.action_space = Discrete(self.actions.num)
        else:
            self.action_space = Box(self.actions.min, self.actions.max, dtype=np.float32)
        # create probability_matrix
        self.probability_matrix = create_matrix(config.grid, self.env_core.env_range)
        self.local_probability_matrix = {i:create_matrix(config.grid, self.env_core.env_range) for i in range(self.agent_num)}
        self.global_matrix_shape = self.probability_matrix.shape
        self.duration_time = int(self._decision_dt / self.env_core.dt) # 决策持续步长
        self.agent_info_shape = config.agent_info_shape
        # ob_space and action space 
        self.max_evader_in_obs = self.config.max_evader_in_obs
        self.max_pursuer_in_obs = self.config.max_pursuer_in_obs
        if self._use_map_obs:
            self.observation_space = {
                'friend_info':Box(low=0, high=1, shape=(self.agent_num, *self.friend_map_shape)),
                'target_info':Box(low=0, high=1, shape=(self.agent_num, *self.target_map_shape)),
                'agent_info':Box(low=0, high=1, shape=(self.agent_num, self.agent_info_shape,)),
                'action_mask': Box(low=0, high=1, shape=(self.agent_num, self.action_space.n,), dtype=bool),
                'agent_mask': Box(low=0, high=1, shape=(self.agent_num, 1,)),
            }
        else:
            self.observation_space = {
                'friend_info': Box(low=0, high=1, shape=(self.agent_num, self.max_pursuer_in_obs, self.agent_info_shape)),
                'target_info': Box(low=0, high=1, shape=(self.agent_num, self.max_evader_in_obs, self.agent_info_shape)),
                'agent_info': Box(low=0, high=1, shape=(self.agent_num, self.agent_info_shape,)),
                'action_mask': Box(low=0, high=1, shape=(self.agent_num, self.action_space.n,), dtype=bool),
                'agent_mask': Box(low=0, high=1, shape=(self.agent_num, 1,)),
                'target_mask': Box(low=0, high=1, shape=(self.agent_num, self.max_evader_in_obs,)),
                'friend_mask': Box(low=0, high=1, shape=(self.agent_num, self.max_pursuer_in_obs,)),                        
            }
        if self._use_matrix:
            self.observation_space['matrix_info'] = Box(low=-1, high=1, shape=(self.agent_num, *self.local_matrix_shape))
        if self.use_global_info:
            if self.use_agent_specific_global_info:
                self.observation_space['global_info'] = Box(low=0, high=1, shape=(self.agent_num, 
                                                        (self.max_pursuer_in_obs+self.max_evader_in_obs+1)*self.agent_info_shape,))
            else:
                self.observation_space['global_info'] = Box(low=0, high=1, shape=(1, 
                                                        (self.env_core._max_evader_num+self.env_core._max_pursuer_num)*self.agent_info_shape,))
        
        self.share_reward = config.share_reward
        self.share_observation_space = deepcopy(self.observation_space)
        self.test = self.env_core.test
        # dynamic change the team_spirit 
        self.use_adaptive_team_spirit = config.use_adaptive_team_spirit
        self.team_spirit = config.team_spirit
        self.team_spirit_min = config.team_spirit_min
        self.team_spirit_max = config.team_spirit_max
        self.team_spirit_increase_episode = config.team_spirit_episode
        self.episode_num = 0
        self.fig = plt.figure(figsize=(10,8))

    def set_render_flag(self):

        self.env_core.render = True 

    def set_stage(self, num):

        """
        0: search
        1: tracking
        2: search and tracking
        """
        self._learning_stage = num 
    
    def get_stage(self):

        return self._learning_stage

    def get_info(self):

        info = edict({})
        info.matrix_shape = self.local_matrix_shape
        info.global_matrix_shape = self.global_matrix_shape
        info.local_friend_map_shape = self.friend_map_shape
        info.local_target_map_shape = self.target_map_shape
        info.agent_shape = self.agent_info_shape
        info.action_type = self.config.action_type 
        info.max_action = self.actions.max_action
        info.action_shape = 1
        info.matrix_gamma = self.gamma 
        info.collision_avoidance = self._collision_avoidance
        info.action_space = self.action_space
        info.action_dim = self.action_space.n
        info.learning_stage = self._learning_stage
        info.use_matrix = self._use_matrix
        # info.global_info_shape = (self.agent_num+self.evader_num, self.agent_info_shape)
        if self.use_agent_specific_global_info:
            info.global_info_shape =  (self.max_pursuer_in_obs+self.max_evader_in_obs+1, self.agent_info_shape)
        else:
            info.global_info_shape = (self.env_core._max_pursuer_num+self.env_core._max_evader_num, self.agent_info_shape)
        info.agent_num = self.agent_num

        full_info = edict({**self.env_info, **info}) 
        return full_info 

    def reset(self, pursuer_num=None, evader_num=None):
        
        if self.use_adaptive_team_spirit:
            self.team_spirit = self.team_spirit_min + (
            (self.team_spirit_max-self.team_spirit_min)* min(self.episode_num / self.team_spirit_increase_episode, 1))
        if self.share_reward:
            self.team_spirit = 1
        self.matrix_save_to_visualize = None
        self.local_matrix = None
        obs_dict = self.env_core.reset(pursuer_num, evader_num)
        self.pursuer_num = len(self.env_core.pursuers)
        self.evader_num = len(self.env_core.evaders)
        # 局部概率矩阵可视化
        self.local_merge_before_matrix_save = []
        self.local_merge_after_matrix_save = []
        self.local_matrix_save = []
        self.probability_matrix = create_matrix(self.config.grid, self.env_core.env_range)
        self.local_probability_matrix = (
        {i:create_matrix(self.config.grid, self.env_core.env_range) for i in range(self.agent_num)})
        self._update_global_probability_matrix(obs_dict)
        self._get_local_matrix(obs_dict)
        if self._use_map_obs:
            obs = self.get_local_map_obs(obs_dict)
        else:
            obs = self.get_local_vector_obs(obs_dict)
        self.pre_obs = deepcopy(obs) 
        self.pre_obs_dict = deepcopy(obs_dict)

        self.episode_num += 1
        
        self.count_map = [[[0,0] for _ in range(int(self.max_env_range // 0.5))] for _ in range(int(self.max_env_range // 0.5))] # （访问次数，上次访问时间）
        self.cur_time = 0
        # 计算初始的不确定性
        self.last_global_certainty_ratio = np.mean(np.abs(self.probability_matrix - 0.5)/0.5) # 
        self.last_local_certainty_ratio = np.mean(np.abs(self.local_matrix[:,-1] - 0.5)/0.5, axis=(1,2)) #

        return obs 

    def compute_tracking_rewards(self, obs_dict, info):
        """
        计算追踪的奖励
        """
        exploring_rewards = np.zeros(self.agent_num)
        for index, pursuer in enumerate(self.env_core.pursuers):
            global_certainty_ratio = np.mean(np.abs(self.probability_matrix - 0.5)/0.5) # 
            # certainty_increase = global_certainty_ratio - last_global_certainty_ratio
            exploring_rewards[index] = global_certainty_ratio 

        tracking_rewards = np.zeros(self.agent_num)
        for index, pursuer in enumerate(self.env_core.pursuers):
            tracking_rewards[index] = min(1, pursuer.find_evader_num)
        collision_rewards = np.zeros(self.agent_num)
        if self._collision_avoidance:
            collision_rewards = self.compute_collision_rewards(info)
            tracking_rewards += collision_rewards
        # tracking_rewards += 0.5*exploring_rewards
        reward_info = {'tracking_rewards': tracking_rewards, 'collision_rewards': collision_rewards, 'explore_rewards': exploring_rewards}
        rewards = self._track_weight*tracking_rewards + self._explore_weight*exploring_rewards
        if self.share_reward:
            rewards = np.mean(rewards)
        return rewards, reward_info
 
    def get_explore_bonus(self):

        explore_bonus = 0
        for pursuer in self.env_core.pursuers:
            pos = pursuer.pos
            x = int(min(pos[0] // 0.5, len(self.count_map)-1))
            y = int(min(pos[1] // 0.5, len(self.count_map[0])-1))
            self.count_map[x][y][0] += 1
            dif_time = self.cur_time - self.count_map[x][y][1]
            self.count_map[x][y][1] = self.cur_time
            if self.count_map[x][y][0] == 1:
                explore_bonus += 0.1
            else:
                explore_bonus += (dif_time / np.sqrt(self.count_map[x][y][0]) / 1000)
        return explore_bonus 

    def compute_exploring_rewards(self, obs_dict, info):
        """
        计算探索的奖励: 1. 对区域了解程度的增加 2. 鼓励尽可能分散
        """
        friend_observe_dict = obs_dict['friend_observe_dict']
        exploring_rewards = np.zeros(self.agent_num)
        global_certainty_ratio = np.mean(np.abs(self.probability_matrix - 0.5)/0.5) # 
        self.last_global_certainty_ratio = global_certainty_ratio
        for index, pursuer in enumerate(self.env_core.pursuers):
            local_certainty_ratio = np.mean(np.abs(self.local_matrix[index][-1] - 0.5)/0.5) # 
            punishment = 0 
            friend_id = friend_observe_dict[index]
            pose = self.env_core.pursuers[index].pos
            for agent_id in friend_id:
                distance = l2norm(pose, self.env_core.pursuers[agent_id].pos)
                if distance < 2*self.env_info.search_radius:
                    punishment += (-0.5*(np.exp((2*self.env_info.search_radius- distance)/2*self.env_info.search_radius)-1))
            exploring_rewards[index] = global_certainty_ratio - local_certainty_ratio + punishment # 增加全局的确定性同时减小局部的确定性

        collision_rewards = np.zeros(self.agent_num)
        if self._collision_avoidance:
            collision_rewards = self.compute_collision_rewards(info)
            exploring_rewards += collision_rewards
        if self.share_reward:
            exploring_rewards = np.mean(exploring_rewards)    
        reward_info = {"global_certainty_ratio":global_certainty_ratio, 
                       'overlapping_punishment':punishment, 
                       'collision_reward': collision_rewards,
                       'local_centainty_ratio': local_certainty_ratio,}
        
        return exploring_rewards, reward_info
    
    def compute_all_rewards(self, info, obs_dict):
        if not self.share_reward:

            """
            探索奖励
            """
            exploring_rewards = np.zeros(self.agent_num)
            overlapping_punishment = np.zeros(self.agent_num)
            friend_observe_dict = obs_dict['friend_observe_dict']
            for index, pursuer in enumerate(self.env_core.pursuers):
                global_certainty_ratio = np.mean(np.abs(self.probability_matrix - 0.5)/0.5) # 
                local_certainty_ratio = np.mean(np.abs(self.local_matrix[index][-1] - 0.5)/0.5) # 
                punishment = 0 
                friend_id = friend_observe_dict[index]
                pose = self.env_core.pursuers[index].pos
                for agent_id in friend_id:
                    distance = l2norm(pose, self.env_core.pursuers[agent_id].pos)
                    if distance < 2*self.env_info.search_radius:
                        punishment += (-0.5*(np.exp((2*self.env_info.search_radius- distance)/2*self.env_info.search_radius)-1))
                overlapping_punishment[index] = punishment
                exploring_rewards[index] = global_certainty_ratio - 0.5*local_certainty_ratio 
            """
            
            团队奖励
            """
            find_evader_num = info['find_evader_num']
            team_rewards = np.ones(self.agent_num)*(find_evader_num / self.env_core.evaders_num)
            '''
            追踪奖励
            '''
            tracking_rewards = np.zeros(self.agent_num)
            for index, pursuer in enumerate(self.env_core.pursuers):
                tracking_rewards[index] = min(1, pursuer.find_evader_num)

            '''
            惩罚重复观测同一个目标
            '''
            repeat_track_punishments = np.zeros(self.agent_num)
            who_observe_evader_dict = obs_dict['who_observe_evader_dict']
            for evader in self.env_core.evaders:
                pursuer_id_list = who_observe_evader_dict[evader.id]
                if len(pursuer_id_list)>1:
                    repeat_track_punishments[pursuer_id_list[1:]] = -1 

            '''
            避碰奖励
            '''
            collision_rewards = np.zeros(self.agent_num)
            if self._collision_avoidance:
                collision_rewards = self.compute_collision_rewards(info)
            # team_spirit from 0 to 1
            final_reward = (self.team_spirit*team_rewards + (1-self.team_spirit)*tracking_rewards)*self._track_weight + \
                            self._explore_weight*exploring_rewards + self._collision_weight*collision_rewards + \
                            self._overlapping_punishment_weight * overlapping_punishment + self._repeat_track_punishment_weight * repeat_track_punishments
            
            reward_info = {'team_rewards': team_rewards, 
                           'explore_rewards': exploring_rewards, 
                           'tracking_rewards': tracking_rewards, 
                           'collision_rewards':collision_rewards,
                           'overlapping_punishment': overlapping_punishment,
                           'repeat_track_punishments': repeat_track_punishments}
        else:

            """
            探索奖励
            """
            exploring_reward = 0
            global_certainty_ratio = np.mean(np.abs(self.probability_matrix - 0.5)/0.5)
            exploring_reward = global_certainty_ratio        
            """
            团队追踪奖励
            """
            team_reward = 0
            find_evader_num = info['find_evader_num']
            team_reward = (find_evader_num / self.env_core.evaders_num)

            '''
            避碰奖励
            '''
            collision_rewards = 0
            if self._collision_avoidance:
                collision_rewards = np.mean(self.compute_collision_rewards(info))
            
            final_reward = self._track_weight*team_reward  + (
            self._explore_weight*exploring_reward + self._collision_weight*collision_rewards)
            reward_info = {'team_rewards': team_reward, 
                           'explore_rewards': exploring_reward, 
                           'collision_rewards': collision_rewards}
        
        return np.array(final_reward), reward_info

    def compute_collision_rewards(self, info):
        
        """
        计算碰撞奖励
        """
        
        collision_rewards = np.zeros(self.agent_num)
        collision_with_pursuer = info['collision_with_pursuer']
        for i in range(self.agent_num):
            if i >= len(self.env_core.pursuers):
                break
            if self._collision_avoidance:
                collision_rewards[i] = -1 if collision_with_pursuer[i] else 0
        return collision_rewards

    def compute_rewards(self, obs_dict, info):
    
        if self._learning_stage == 0:
            return self.compute_exploring_rewards(obs_dict, info)
        elif self._learning_stage == 1:
            return self.compute_tracking_rewards(obs_dict, info)
        elif self._learning_stage == 2: # 分层训练
            return self.compute_all_rewards(info, obs_dict)
        elif self._learning_stage == 3: # 端到端训练
            return self.compute_all_rewards(info, obs_dict)
    
    def _actions_transform(self, actions):
        
        all_action = []
        for action in actions:
            if self.config.action_type == 'discrete':
                real_action = self.actions.map[action]
            else:
                real_action = action
            all_action.append(real_action)
        return all_action
    
    def step(self, actions):

        self.cur_time += 1
        if isinstance(self.action_space, Discrete):
            actions = np.array(actions).reshape(-1)
        actions = self._actions_transform(actions)
        for _ in range(self.duration_time):
            obs_dict, done, info = self.env_core.step(actions)
            if done:
                break
        self._update_global_probability_matrix(obs_dict)
        self._get_local_matrix(obs_dict)
        if self._use_map_obs:
            obs = self.get_local_map_obs(obs_dict)
        else:
            obs = self.get_local_vector_obs(obs_dict)
        rewards, rewards_info = self.compute_rewards(obs_dict, info)
        if done:
            info = info['metrix']
        self.pre_obs = deepcopy(obs)
        
        return obs, rewards, done, info, rewards_info

    def draw_traj_2d(self, ax, agents_info, agents_traj_list):
        
        for idx, agent in enumerate(agents_info):

            agent_gp = agent.group
            agent_rd = agent.radius
            sec_radius = agent.sec_radius
            comm_radius = agent.comm_radius

            if agent_gp == 0:
                plt_color = [0,0,1]
            elif agent_gp == 1:
                plt_color = [0, 0, 0]
            history_pos = np.array(agents_traj_list[idx])
            pos_x = history_pos[:, 0]
            pos_y = history_pos[:, 1]
            if agent_gp == 0:
                cir1 = Circle(xy = (pos_x[-1], pos_y[-1]), radius=sec_radius, alpha=0.2)
                cir2 = Circle(xy = (pos_x[-1], pos_y[-1]), radius=comm_radius, alpha=0.1)
                ax.add_patch(cir1)
                ax.add_patch(cir2)
            # 绘制渐变线（保留最近200步的轨迹）
            colors = np.zeros((min(len(pos_x), 200), 4))
            colors[:, :3] = plt_color
            colors[:, 3] = np.linspace(0.1, 1., min(len(pos_x), 200), endpoint=True)
            colors = rgba2rgb(colors)
            ax.scatter(pos_x[-200:],  # 绘制轨迹
                        pos_y[-200:], 
                        color=colors, s=3, alpha=0.5)
            if agent_gp == 0:
                my_model = get_2d_uav_model(size=agent_rd*10)
                color = 'blue'
            else:
                my_model = get_2d_car_model(size=agent_rd*10)
                color = 'black'
            pos = [pos_x[-1], pos_y[-1]]
            heading = agent.heading
            draw_agent_2d(ax, pos, heading, my_model, color)
            
    def _get_local_matrix(self, obs_dict, method='traverse'):
        
        if self.use_global_matrix:
            for i in range(self.agent_num):
                self.local_probability_matrix[i] = deepcopy(self.probability_matrix)
        else:
            self._update_local_probability_matrix(obs_dict)
        all_local_matrix = np.zeros((self.agent_num, *self.local_matrix_shape[1:]))
        if method == 'pad':
            ratio = int(self.matrix_grid // self.local_matrix_grid)
            global_shape = ratio* np.array(self.global_matrix_shape)
            global_matrix_pad = np.zeros(global_shape)
            for i in range(self.global_matrix_shape[0]):
                for j in range(self.global_matrix_shape[1]):
                    global_matrix_pad[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio] = self.probability_matrix[i][j]
            global_matrix_pad = np.pad(global_matrix_pad, int(self.local_matrix_shape[1]//2))
            for idx in range(self.agent_num):
                agent_pos = self.env_core.pursuers[idx].pos
                center_x = agent_pos[0] // self.local_matrix_grid + int(self.local_matrix_shape[1]//2)
                center_y = agent_pos[1] // self.local_matrix_grid + int(self.local_matrix_shape[2]//2)
                right_x = min(int(center_x + self.local_matrix_shape[1]//2), global_matrix_pad.shape[0])
                left_x = max(int(center_x - self.local_matrix_shape[1]//2), 0)
                top_y = min(int(center_y + self.local_matrix_shape[2]//2), global_matrix_pad.shape[1]) 
                bottom_y = max(int(center_y - self.local_matrix_shape[2]//2), 0)
                local_matrix = deepcopy(global_matrix_pad[left_x:right_x, bottom_y:top_y])
                all_local_matrix[idx] = local_matrix
        elif method == 'traverse':
            for index in range(self.agent_num):
                if index >= len(self.env_core.pursuers):
                    break
                local_matrix = np.zeros(self.local_matrix_shape[1:])
                for i in range(self.local_matrix_shape[1]):
                    for j in range(self.local_matrix_shape[2]):
                        agent_pos = self.env_core.pursuers[index].pos
                        local_x = (i-self.local_matrix_shape[1]//2)*self.local_matrix_grid + agent_pos[0]
                        local_y = (j-self.local_matrix_shape[2]//2)*self.local_matrix_grid + agent_pos[1]
                        if local_x < 0 or local_x > self.max_env_range or local_y < 0 or local_y > self.max_env_range:
                            continue
                        pos = np.clip([local_x, local_y], 0, self.max_env_range-0.01)
                        global_x = int(pos[0] // self.matrix_grid)
                        global_y = int(pos[1] // self.matrix_grid)
                        local_matrix[i][j] = (self.local_probability_matrix[index])[global_x][global_y]

                all_local_matrix[index] = local_matrix
        if self.local_matrix is None: 
            self.local_matrix = all_local_matrix[:, None,].repeat(self.local_matrix_shape[0], axis=1)
        else:
            self.local_matrix = np.delete(self.local_matrix, 0, axis=1)
            self.local_matrix = np.append(self.local_matrix, all_local_matrix[:, None,], axis=1)
        if self._save_matrix:
            self.local_matrix_save.append(all_local_matrix)
                    
    def get_local_vector_obs(self, obs_dict):
        
        """
        得到向量化的观测信息
        """
        # 向量观测信息
        friend_observe_dict = obs_dict['friend_observe_dict']
        pursuer_observe_evader_dict = obs_dict['pursuer_observe_evader_dict']
        agent_vector = np.zeros((self.agent_num, self.agent_info_shape))
        all_friend_vector = np.zeros((self.agent_num, self.max_pursuer_in_obs, self.agent_info_shape))
        all_target_vector = np.zeros((self.agent_num, self.max_evader_in_obs, self.agent_info_shape))
        target_mask = np.zeros((self.agent_num, self.max_evader_in_obs)) # 屏蔽无效的目标信息 0 表示无效
        friend_mask = np.zeros((self.agent_num, self.max_pursuer_in_obs)) # 屏蔽无效的队友信息
        agent_mask = np.zeros((self.agent_num, 1))
        action_mask = np.zeros((self.agent_num, self.action_space.n))
        for index in range(self.agent_num):
            # 计算概率矩阵的质心
            centroid = np.array([0., 0.])
            total_weight = 0
            local_matrix = np.mean(self.local_matrix[index], axis=0)
            for i in range(local_matrix.shape[0]):
                for j in range(local_matrix.shape[1]):
                    weight = 1 / (np.abs(local_matrix[i,j] - 0.5) / 0.5 + 1)
                    centroid[0] += weight * i
                    centroid[1] += weight * j
                    total_weight += weight
            if total_weight != 0:
                centroid /= total_weight
            explore_vector = centroid - np.array([(local_matrix.shape[0]-1) / 2 , (local_matrix.shape[1]-1) / 2])
            if index >= len(self.env_core.pursuers):
                break
            agent_mask[index] = 1 # 置有效位
            agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
            agent_vector[index, :2] = agent_pos
            agent_vector[index, 2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
            agent_vector[index, 3] = self.env_core.pursuers[index].heading / (np.pi*2)
            # agent_vector[index, 4:6] = explore_vector
            
            friend_info = friend_observe_dict[index]
            friend_dis_record = {}
            # 按距离由近及远排序
            for j, agent_id in enumerate(friend_info):
                target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                distance = l2norm(agent_pos, target_pos)
                friend_dis_record[agent_id] = distance
            sorted_list = sorted(friend_dis_record.items(), key=lambda x:x[1])
            friend_mask[index, :min(len(friend_info), self.max_pursuer_in_obs)] = 1
            for i, pair in enumerate(sorted_list):
                agent_id = pair[0]
                if i >= self.max_pursuer_in_obs:
                    break
                friend_pos = self.env_core.pursuers[agent_id].pos / self.max_env_range
                friend_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                friend_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                all_friend_vector[index, i, :2] = friend_pos
                all_friend_vector[index, i, 2] = friend_v
                all_friend_vector[index, i, 3] = friend_heading
            
            target_info = pursuer_observe_evader_dict[index]
            if self._learning_stage == 2:
                action_mask[index, min(self.max_evader_in_obs, len(target_info))+2:] = 1
                if len(target_info) >= 1:
                    action_mask[index, 1] = 1
            # learning_stage == 1 is tracking stage , only consider the near target 
            if self._learning_stage == 1:
                target_mask[index, :min(len(target_info),1)] = 1
            else:
                target_mask[index, :min(len(target_info), self.max_evader_in_obs)] = 1
            target_dis_record = {} 
            # 按距离由近及远排序
            for j, agent_id in enumerate(target_info):
                target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                distance = l2norm(agent_pos, target_pos)
                target_dis_record[agent_id] = distance
            sorted_list = sorted(target_dis_record.items(), key=lambda x:x[1])
            for i, pair in enumerate(sorted_list):
                if i >= self.max_evader_in_obs:
                    break
                agent_id = pair[0]
                target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                target_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                target_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                all_target_vector[index, i, :2] = target_pos
                all_target_vector[index, i, 2:4] = [target_v, target_heading]
        observation = { 'friend_info': all_friend_vector.copy(),
                        'target_info': all_target_vector.copy(),
                        'agent_info': agent_vector.copy(),
                        'target_mask': target_mask.copy(),
                        'friend_mask': friend_mask.copy(),
                        'agent_mask': agent_mask.copy(),
                        'action_mask': action_mask.copy(),}
        # 概率矩阵信息
        if self._use_matrix:
            observation.update({'matrix_info':self.local_matrix.copy()})
        
        # 全局共享观测信息
        if self.use_global_info:
            if self.use_agent_specific_global_info:
                
                all_friend_vector = np.zeros((self.agent_num, self.max_pursuer_in_obs, self.agent_info_shape))
                all_target_vector = np.zeros((self.agent_num, self.max_evader_in_obs, self.agent_info_shape))
                
                for index in range(self.agent_num):
                    if index >= len(self.env_core.pursuers):
                        break
                    agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
                    # agent_vector[index, :2] = agent_pos
                    # agent_vector[index, 2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
                    # agent_vector[index, 3] = self.env_core.pursuers[index].heading / (np.pi*2)
                    
                    friend_info = [i for i in range(len(self.env_core.pursuers))]
                    friend_dis_record = {}
                    # 按距离由近及远排序
                    for j, agent_id in enumerate(friend_info):
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        distance = l2norm(agent_pos, target_pos)
                        friend_dis_record[agent_id] = distance
                    sorted_list = sorted(friend_dis_record.items(), key=lambda x:x[1])
                    for i, pair in enumerate(sorted_list):
                        agent_id = pair[0]
                        if agent_id == index:
                            break
                        if i >= self.max_pursuer_in_obs:
                            break
                        friend_pos = self.env_core.pursuers[agent_id].pos / self.max_env_range
                        friend_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                        friend_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                        all_friend_vector[index, i, :2] = friend_pos
                        all_friend_vector[index, i, 2] = friend_v
                        all_friend_vector[index, i, 3] = friend_heading
                        
                    target_dis_record = {} 
                    target_info = [j+len(self.env_core.pursuers) for j in range(len(self.env_core.evaders))]
                    # 按距离由近及远排序
                    for j, agent_id in enumerate(target_info):
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        distance = l2norm(agent_pos, target_pos)
                        target_dis_record[agent_id] = distance
                    sorted_list = sorted(target_dis_record.items(), key=lambda x:x[1])
                    for i, pair in enumerate(sorted_list):
                        if i >= self.max_evader_in_obs:
                            break
                        agent_id = pair[0]
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        target_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                        target_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                        all_target_vector[index, i, :2] = target_pos
                        all_target_vector[index, i, 2:4] = [target_v, target_heading]
                global_info = np.concatenate(
                            [agent_vector[:,None,], all_friend_vector, all_target_vector], axis=1).reshape(self.agent_num, -1)
                observation['global_info'] = global_info.copy()

            else:
                target_id_list = obs_dict['pursuers_observing_list']
                all_friend_vector = np.zeros((self.env_core._max_pursuer_num, self.agent_info_shape))
                all_target_vector = np.zeros((self.env_core._max_evader_num, self.agent_info_shape))
                for i in range(len(self.env_core.pursuers)):
                    agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
                    all_friend_vector[i][:2] = agent_pos
                    all_friend_vector[i][2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
                    all_friend_vector[i][3] = self.env_core.pursuers[index].heading / (np.pi*2)
                for i, target_id in enumerate(target_id_list):
                    agent_pos = self.env_core.agents[target_id].pos / self.max_env_range
                    all_target_vector[i][:2] = agent_pos
                    all_target_vector[i][2] = self.env_core.agents[target_id].pref_speed if not self._ignore_speed else 1
                    all_target_vector[i][3] = self.env_core.agents[target_id].heading / (np.pi*2)
                global_info = np.concatenate(
                                [all_friend_vector[None,:], all_target_vector[None,:]], axis=1).reshape(1,-1)
                observation['global_info'] = global_info.copy()
        return observation  
    
    def get_local_map_obs(self, obs_dict):
        
        """
            得到图表征的观测信息
        """
        
        friend_observe_dict = obs_dict['friend_observe_dict']
        pursuer_observe_evader_dict = obs_dict['pursuer_observe_evader_dict']
        all_friend_map = np.zeros((self.agent_num, *self.friend_map_shape))
        all_target_map = np.zeros((self.agent_num, *self.target_map_shape))
        all_local_matrix = np.zeros((self.agent_num, *self.local_matrix_shape))
        agent_vector = np.zeros((self.agent_num, self.agent_info_shape))
        agent_mask = np.zeros((self.agent_num, 1))
        action_mask = np.zeros((self.agent_num, self.action_space.n))
        for index in range(self.agent_num):
            if index >= len(self.env_core.pursuers):
                break
            agent_mask[index] = 1 # 置有效位
            agent_pos = self.env_core.pursuers[index].pos
            agent_vector[index, 0] = agent_pos[0] / self.max_env_range
            agent_vector[index, 1] = agent_pos[1] / self.max_env_range
            agent_vector[index, 2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
            agent_vector[index, 3] = self.env_core.pursuers[index].heading
            friend_info = friend_observe_dict[index]
            target_info = pursuer_observe_evader_dict[index]
            local_friend_matrix = np.zeros(self.friend_map_shape)
            local_target_matrix = np.zeros(self.target_map_shape)
            target_matrix_grid = self.target_map_grid
            friend_matrix_grid = self.friend_map_grid
            for agent_id in friend_info:
                friend_pos = self.env_core.pursuers[agent_id].pos
                friend_v = self.env_core.agents[agent_id].pref_speed
                friend_heading = self.env_core.agents[agent_id].heading
                offset_x = int((friend_pos[0] - agent_pos[0]) // friend_matrix_grid) 
                offset_y = int((friend_pos[1] - agent_pos[1]) // friend_matrix_grid)
                friend_x = self.friend_map_shape[-2]//2-1+offset_x
                friend_y = self.friend_map_shape[-1]//2-1+offset_y
                if friend_x< 0 or friend_x>self.friend_map_shape[-2]-1 or (
                friend_y< 0 or friend_y>self.friend_map_shape[-1]-1):
                    continue
                local_friend_matrix[0, friend_x, friend_y] += 1
                num = local_friend_matrix[0, friend_x, friend_y]
                pos_x = local_friend_matrix[1, friend_x, friend_y]
                pos_y = local_friend_matrix[2, friend_x, friend_y]
                heading = local_friend_matrix[4, friend_x, friend_y]
                local_friend_matrix[1, friend_x, friend_y] = (num-1/(num))*pos_x + agent_pos[0] / self.max_env_range / num
                local_friend_matrix[2, friend_x, friend_y] = (num-1/(num))*pos_y + agent_pos[1] / self.max_env_range / num
                local_friend_matrix[3, friend_x, friend_y] = friend_v if not self._ignore_speed else 1
                local_friend_matrix[4, friend_x, friend_y] = (num-1/(num))*heading + friend_heading/num
            for agent_id in target_info:
                target_pos = self.env_core.agents[agent_id].pos
                target_v = self.env_core.agents[agent_id].pref_speed
                target_heading = self.env_core.agents[agent_id].heading
                offset_x = int((target_pos[0] - agent_pos[0]) // target_matrix_grid)
                offset_y = int((target_pos[1] - agent_pos[1]) // target_matrix_grid)
                target_x = self.target_map_shape[-1]//2 - 1 + offset_x
                target_y = self.target_map_shape[-2]//2 - 1 + offset_y
                if target_x< 0 or target_x>self.target_map_shape[-2]-1 or (
                target_y< 0 or target_y>self.target_map_shape[-1]-1):
                    continue
                local_target_matrix[0, target_x, target_y] += 1
                num = local_target_matrix[0, target_x, target_y]
                pos_x = local_target_matrix[1, target_x, target_y]
                pos_y = local_target_matrix[2, target_x, target_y]
                heading = local_target_matrix[4, target_x, target_y]
                local_target_matrix[1, target_x, target_y] = (num-1/(num))*pos_x + agent_pos[0] / self.max_env_range / num
                local_target_matrix[2, target_x, target_y] = (num-1/(num))*pos_y + agent_pos[1] / self.max_env_range / num
                local_target_matrix[3, target_x, target_y] = target_v if not self._ignore_speed else 1
                local_target_matrix[4, target_x, target_y] = (num-1/(num))*heading + target_heading / num

            local_matrix = np.ones(self.local_matrix_shape, dtype=np.float32)*(-1)
            origin = (int(agent_pos[0] // self.matrix_grid)-3, int(agent_pos[1] // self.matrix_grid)-3)
            for i in range(local_matrix.shape[1]):
                if i+origin[0] < 0 or i+origin[0] > self.local_probability_matrix[index].shape[0]-1 :
                    continue
                for j in range(local_matrix.shape[2]):
                    if j+origin[1] < 0 or j+origin[1] > self.local_probability_matrix[index].shape[0]-1:
                        continue
                    local_matrix[0][i][j] = self.local_probability_matrix[index][i+origin[0]][j+origin[1]]
            all_local_matrix[index] = local_matrix
            all_friend_map[index] = local_friend_matrix
            all_target_map[index] = local_target_matrix

        # 全局共享观测信息
        if self.use_global_info:
            if self.use_agent_specific_global_info:
                
                all_friend_vector = np.zeros((self.agent_num, self.max_pursuer_in_obs, self.agent_info_shape))
                all_target_vector = np.zeros((self.agent_num, self.max_evader_in_obs, self.agent_info_shape))
                
                for index in range(self.agent_num):
                    if index >= len(self.env_core.pursuers):
                        break
                    agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
                    
                    friend_info = [i for i in range(len(self.env_core.pursuers))]
                    friend_dis_record = {}
                    # 按距离由近及远排序
                    for j, agent_id in enumerate(friend_info):
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        distance = l2norm(agent_pos, target_pos)
                        friend_dis_record[agent_id] = distance
                    sorted_list = sorted(friend_dis_record.items(), key=lambda x:x[1])
                    for i, pair in enumerate(sorted_list):
                        agent_id = pair[0]
                        if agent_id == index:
                            break
                        if i >= self.max_pursuer_in_obs:
                            break
                        friend_pos = self.env_core.pursuers[agent_id].pos / self.max_env_range
                        friend_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                        friend_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                        all_friend_vector[index, i, :2] = friend_pos
                        all_friend_vector[index, i, 2] = friend_v
                        all_friend_vector[index, i, 3] = friend_heading
                        
                    target_dis_record = {} 
                    target_info = [j+len(self.env_core.pursuers) for j in range(len(self.env_core.evaders))]
                    # 按距离由近及远排序
                    for j, agent_id in enumerate(target_info):
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        distance = l2norm(agent_pos, target_pos)
                        target_dis_record[agent_id] = distance
                    sorted_list = sorted(target_dis_record.items(), key=lambda x:x[1])
                    for i, pair in enumerate(sorted_list):
                        if i >= self.max_evader_in_obs:
                            break
                        agent_id = pair[0]
                        target_pos = self.env_core.agents[agent_id].pos / self.max_env_range
                        target_v = self.env_core.agents[agent_id].pref_speed if not self._ignore_speed else 1
                        target_heading = self.env_core.agents[agent_id].heading / (np.pi*2)
                        all_target_vector[index, i, :2] = target_pos
                        all_target_vector[index, i, 2:4] = [target_v, target_heading]
                global_info = np.concatenate(
                            [agent_vector[:,None,], all_friend_vector, all_target_vector], axis=1).reshape(self.agent_num, -1)

            else:
                target_id_list = obs_dict['pursuers_observing_list']
                all_friend_vector = np.zeros((self.env_core._max_pursuer_num, self.agent_info_shape))
                all_target_vector = np.zeros((self.env_core._max_evader_num, self.agent_info_shape))
                for i in range(len(self.env_core.pursuers)):
                    agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
                    all_friend_vector[i][:2] = agent_pos
                    all_friend_vector[i][2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
                    all_friend_vector[i][3] = self.env_core.pursuers[index].heading / (np.pi*2)
                for i, target_id in enumerate(target_id_list):
                    agent_pos = self.env_core.agents[target_id].pos / self.max_env_range
                    all_target_vector[i][:2] = agent_pos
                    all_target_vector[i][2] = self.env_core.agents[target_id].pref_speed if not self._ignore_speed else 1
                    all_target_vector[i][3] = self.env_core.agents[target_id].heading / (np.pi*2)
                global_info = np.concatenate(
                                [all_friend_vector[None,:], all_target_vector[None,:]], axis=1).reshape(1,-1)

        return {'friend_info': all_friend_map,
                'target_info': all_target_map,
                'matrix_info': all_local_matrix,
                'agent_info': agent_vector,
                'agent_mask': agent_mask,
                'action_mask': action_mask,
                'global_info': global_info.copy()}
        
    def _update_global_probability_matrix(self, obs_dict):
        
        """
        更新全局概率矩阵
        """
        pursur_list = self.env_core.pursuers
        evader_list = [self.env_core.agents[id] for id in obs_dict['pursuers_observing_list']]
        update_probability_matrix(self.max_env_range, self.gamma, 
                                  self.matrix_grid, pursur_list, 
                                  evader_list, self.probability_matrix, method=self._matrix_computation)
        if self.matrix_save_to_visualize is None:
            self.matrix_save_to_visualize = np.array([self.probability_matrix])
        else:
            self.matrix_save_to_visualize = np.vstack([self.matrix_save_to_visualize,
                                                    np.array([self.probability_matrix])])

    def _update_local_probability_matrix(self, obs_dict):

        """
        更新局部概率矩阵
        """
        for index in self.local_probability_matrix.keys():
            if index >= len(self.env_core.pursuers):
                break
            pursur_list = [self.env_core.pursuers[id] for id in obs_dict['friend_observe_dict'][index]]
            pursur_list.append(self.env_core.pursuers[index])
            evader_list = [self.env_core.agents[id] for id in obs_dict['pursuer_observe_evader_dict'][index]]
            update_probability_matrix(self.max_env_range, self.gamma, 
                                      self.matrix_grid, pursur_list, evader_list, 
                                      self.local_probability_matrix[index], method=self._matrix_computation)
        

        if self._save_matrix:
            self.local_merge_before_matrix_save.append(np.array([self.local_probability_matrix[i] for i in range(self.agent_num)]))
        """
        融合局部概率矩阵
        """
        self.local_probability_matrix = merge_local_probability_matrix(obs_dict, self.local_probability_matrix)

        if self._save_matrix:
            self.local_merge_after_matrix_save.append(np.array([self.local_probability_matrix[i] for i in range(self.agent_num)]))

    def save_info(self, log_save_dir=None):
        if log_save_dir is None:
            log_save_dir = str(Path((os.path.dirname(os.path.realpath(__file__)))) / 'log')
        # 扩充维度
        self.matrix_save_to_visualize = self.matrix_save_to_visualize[:,None,:,:].repeat(self.duration_time, 1)
        self.matrix_save_to_visualize = self.matrix_save_to_visualize.reshape(-1, *self.global_matrix_shape)
        np.save(log_save_dir + '/matrix', self.matrix_save_to_visualize)
        if self._save_matrix:
            self.local_merge_before_matrix_save = np.array(self.local_merge_before_matrix_save)[:,None,:,:].repeat(self.duration_time, 1)
            self.local_merge_before_matrix_save = self.local_merge_before_matrix_save.reshape(-1, self.agent_num, *self.global_matrix_shape)
            self.local_merge_after_matrix_save = np.array(self.local_merge_after_matrix_save)[:,None,:,:].repeat(self.duration_time, 1)
            self.local_merge_after_matrix_save = self.local_merge_after_matrix_save.reshape(-1, self.agent_num, *self.global_matrix_shape)
            self.local_matrix_save = np.array(self.local_matrix_save)[:,None,:,:,:].repeat(self.duration_time, 1)
            self.local_matrix_save = self.local_matrix_save.reshape(-1, self.agent_num, *self.local_matrix_shape[1:])
            np.save(log_save_dir + f'/local_matrix', np.array(self.local_merge_before_matrix_save))
            np.save(log_save_dir + f'/local_merge_matrix', np.array(self.local_merge_after_matrix_save))
            np.save(log_save_dir + f'/local_agent_matrix', self.local_matrix_save)
        # trajectory
        writer = pd.ExcelWriter(log_save_dir + '/trajs.xlsx')
        for agent in self.env_core.agents:
            data = pd.DataFrame(agent.history_info)
            data.to_excel(writer, sheet_name='agent' + str(agent.id))
        writer._save()
        # scenario information
        info_dict_to_visualize = {
            'all_agent_info': [],
            'all_obstacle': [],
            'some_config_info': {},
        }
        for agent in self.env_core.agents:
            agent_info_dict = {'id': agent.id, 'gp': agent.group, 'radius': agent.radius, 
                               'sec_radius': agent.sec_radius, 'commu_radius': agent.comm_radius}
            info_dict_to_visualize['all_agent_info'].append(agent_info_dict)
        info_dict_to_visualize['some_config_info'] = {'grid': self.matrix_grid,
                                                      'matrix_gamma': self.gamma,
                                                      'env_range': self.env_info.env_range,
                                                      'agent_num': self.agent_num,
                                                      'evader_num': self.evader_num,
                                                      'pursuer_num': self.pursuer_num,
                                                      'dt':self.env_core.dt,}
        info_str = json.dumps(info_dict_to_visualize, indent=4)
        with open(log_save_dir + '/env_cfg.json', 'w') as json_file:
            json_file.write(info_str)
        json_file.close()




