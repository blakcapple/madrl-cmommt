"""
Base Environment 
Written by Ruan Yudi 2022.5.25 
"""

import gym
from easydict import EasyDict as edict
from satenv.env_utils.start import init_agent
import numpy as np 
import itertools
from satenv.env_utils.utils import l2norm
from satenv.configs.base_cfg import BaseConfig

class BaseEnv(gym.Env):

    def __init__(self, env_config: BaseConfig):

        self._render = False # 是否可视化
        self._max_length = env_config.total_steps
        self._dt = env_config.dt
        self._search_radius = env_config.search_radius
        self._comm_radius = env_config.comm_radius 
        self._max_pursuer_num = env_config.max_pursuer_num
        self._min_pursuer_num = env_config.min_pursuer_num
        self._min_evader_num = env_config.min_evader_num
        self._max_evader_num = env_config.max_evader_num
        self._pursuer_speed = env_config.pursuer_speed 
        self._evader_speed = env_config.evader_speed
        self._max_angular_v = env_config.max_angular_v 
        self._env_range = env_config.env_range
        self._evader_sec_radius = env_config.evader_sec_radius
        self._evader_policy = env_config.evader_policy
        self._pursuer_policy = env_config.pursuer_policy
        self._local_grid = env_config.local_grid
        self._test = env_config.test 
        self._config = env_config
    
    @property
    def search_radius(self):
        return self._search_radius
    
    @property
    def local_grid(self):
        return self._local_grid
    
    @property
    def pursuers(self):
        return self._pursuers
    
    @property
    def evaders(self):
        return self._evaders
    
    @property
    def render(self):
        return self._render

    @property
    def env_range(self):
        return self._env_range
    
    @property
    def dt(self):
        return self._dt
    
    @property
    def test(self):
        return self._test
    
    @property
    def max_angular_v(self):
        return self._max_angular_v
    
    @property
    def evaders_num(self):
        return self._evaders_num
    
    @property
    def pursuers_num(self):
        return self._pursuers_num
    
    @property
    def agents(self):
        return self._agents
    
    @render.setter
    def render(self, value):
        assert isinstance(value, bool)
        self._render = value

    
    def get_info(self):

        """
        acquire necessary info from env
        """
        env_info = edict({}) 
        env_info.search_radius = self._search_radius
        env_info.env_range = self._env_range
        env_info.max_pursuer_num = self._max_pursuer_num
        env_info.min_pursuer_num = self._min_pursuer_num
        env_info.max_evader_num = self._max_evader_num
        env_info.min_evader_num = self._min_evader_num
        env_info.dt = self._dt
        env_info.pursuer_speed = self._pursuer_speed
        env_info.evader_speed = self._evader_speed
        env_info.evader_sec_radius = self._evader_sec_radius
        env_info.evader_policy = self._evader_policy
        env_info.pursuer_policy = self._pursuer_policy
        env_info.comm_radius = self._comm_radius
        
        return env_info 
    
    def set_seed(self, seed):

        np.random.seed(seed)

    def reset(self, pursuer_num=None, evader_num=None):

        self.out_of_time = False # 超时标志位
        self.env_step = 0 # 环境步数统计(每回合重置)
        self._init_agents(pursuer_num, evader_num)
        self.observed_evader_record = {evader.id:np.zeros(self._max_length) for evader in self._evaders}
        self.observed_env_record = np.zeros((int(self.env_range[0][1]/self._local_grid), int(self.env_range[1][1]/self._local_grid)))
        self.collision_pursuer_record = {pursuer.id:np.zeros(self._max_length) for pursuer in self._pursuers}
        self.obs_dict = {
            'evader_observe_pursuer_dict': {evader.id: [] for evader in self._evaders}, # 逃避者观察到的追逐者的信息
            'pursuer_observe_evader_dict': {pursuer.id: [] for pursuer in self._pursuers}, #  追逐者观潮到的逃避者的信息
            'pursuers_observing_list': [], # 所有追逐者观测到的逃避者的列表
            'friend_observe_dict': {pursuer.id: [] for pursuer in self._pursuers}, # 追逐者观察到的同伴
            'who_observe_evader_dict':{evader.id: [] for evader in self._evaders}, # 每个逃避者被哪个逃避者观察到了
        }
        self.obs_dict = self.get_obs()
            
        return self.obs_dict
    
    def _update_observed_env_record(self, agent_pos, env_step):
        
        origin_x = int(agent_pos[0] // self._local_grid)
        origin_y = int(agent_pos[1] // self._local_grid)
        r =  int(self._search_radius // self._local_grid)
        for i in range(origin_x-r, origin_x+r+1):
            for j in range(origin_y-r, origin_y+r+1):
                if i < 0 or i >= self.observed_env_record.shape[0] or j < 0 or j >= self.observed_env_record.shape[1]:
                    continue
                x = i * self._local_grid
                y = j * self._local_grid
                if (x - agent_pos[0])**2 + (y - agent_pos[1])**2 <= self.search_radius**2:
                    self.observed_env_record[i][j] = env_step
                    if j - 1 >= 0:
                        self.observed_env_record[i][j-1] = env_step
                    if i - 1 >= 0:
                        self.observed_env_record[i-1][j] = env_step
                    if i - 1 >= 0 and j - 1 >= 0:   
                        self.observed_env_record[i-1][j-1] = env_step

    def get_obs(self):

        evader_observe_pursuer_dict = {evader.id: [] for evader in self._evaders}
        pursuer_observe_evader_dict = {pursuer.id: [] for pursuer in self._pursuers}
        pursuers_observing_list = []
        friend_observe_dict = {pursuer.id: [] for pursuer in self._pursuers} 
        who_observe_evader_dict = self.obs_dict['who_observe_evader_dict']

        for pursuer in self._pursuers:
            host_agent_position = pursuer.pos
            self._update_observed_env_record(host_agent_position, self.env_step)
            pursuer.find_evader_num = 0
            for evader in self._evaders :
                evader_position = evader.pos
                distance = l2norm(host_agent_position, evader_position)
                if distance < evader.sec_radius:  # 追逐者在目标的视野内
                    evader_observe_pursuer_dict[evader.id].append(pursuer.id)
                if distance < pursuer.sec_radius:  # 目标在追逐者视野内
                    pursuer.find_evader_num += 1
                    pursuer_observe_evader_dict[pursuer.id].append(evader.id)
                    self.observed_evader_record[evader.id][self.env_step] = 1
                    if evader.id not in pursuers_observing_list:
                        pursuers_observing_list.append(evader.id)
                    if pursuer.id not in who_observe_evader_dict[evader.id]:
                        who_observe_evader_dict[evader.id].append(pursuer.id)
                else: # 目标不在追逐者视野内，从字典里面删除
                    if pursuer.id in who_observe_evader_dict[evader.id]:
                        who_observe_evader_dict[evader.id].remove(pursuer.id)

            for friend in self._pursuers:
                if not friend.id == pursuer.id:
                    friend_position = friend.pos
                    distance = l2norm(host_agent_position, friend_position)
                    if distance < pursuer.comm_radius:  # 通信范围内
                        friend_observe_dict[pursuer.id].append(friend.id)
                        
        # 将通信距离内的队友观察到的目标也加入智能体观测的目标列表中
        for pursuer in self._pursuers:
            for friend_id in friend_observe_dict[pursuer.id]:
                for target_id in pursuer_observe_evader_dict[friend_id]:
                    if target_id not in pursuer_observe_evader_dict[pursuer.id]:
                        pursuer_observe_evader_dict[pursuer.id].append(target_id)

        for evader_id, pursuer_ids in who_observe_evader_dict.items():
            # 对于新加入的id 按照距离由远及近排序
            evader_pos = self._agents[evader_id].pos
            distance_record = {}
            for id in pursuer_ids:
                if id not in self.obs_dict['who_observe_evader_dict'][evader_id]:
                    distance = l2norm(self._pursuers[id].pos, evader_pos)
                    distance_record[id] = distance
            sorted_list = sorted(distance_record.items(), key=lambda x:x[1])
            for index, pair in enumerate(sorted_list):
                who_observe_evader_dict[evader_id][-len(sorted_list)+index] = int(pair[0])
                    
        obs_dict = {
            'evader_observe_pursuer_dict': evader_observe_pursuer_dict, # 逃避者观察到的追逐者的信息
            'pursuer_observe_evader_dict': pursuer_observe_evader_dict, #  追逐者观察到的逃避者的信息
            'pursuers_observing_list': pursuers_observing_list, # 所有追逐者观测到的逃避者的列表
            'friend_observe_dict': friend_observe_dict, # 追逐观察到的同伴
            'who_observe_evader_dict':who_observe_evader_dict, # 谁观察到了逃避者
        }

        return obs_dict

    def _init_agents(self, pursuer_num=None, evader_num=None):
        
        if evader_num is None:
            self._evaders_num = np.random.randint(self._min_evader_num, self._max_evader_num+1)
        if pursuer_num is None:
            self._pursuers_num = np.random.randint(self._min_pursuer_num, self._max_pursuer_num+1)
        pursuer, evader = init_agent(self._config, render=self._render, 
                                     pursuer_num=self._pursuers_num, 
                                     evader_num=self._evaders_num)
        self._pursuers = pursuer
        self._evaders = evader 
        self._agents = pursuer + evader 

    def _take_actions(self, actions, obs_dict):

        all_actions = np.zeros((len(self._agents),2), dtype=np.float32)

        evader_observe_pursuer_dict = obs_dict['evader_observe_pursuer_dict']

        for agent_index, agent in enumerate(self._agents):
            
            if agent.group == 0:
                all_actions[agent_index] = [self._pursuer_speed, actions[agent_index]]
            elif agent.group == 1:
                dict_comm = {
                    'agents': self._agents,
                    'evader_observe_pursuer_dict': evader_observe_pursuer_dict,
                }
                all_actions[agent_index] = agent.find_next_action(dict_comm)

        for i, agent in enumerate(self._agents):
            agent.take_action(all_actions[i], self._dt)

    def step(self, actions):

        self._take_actions(actions, self.obs_dict)
        self.obs_dict = self.get_obs()
        collision_with_pursuer, dist_btwn_nearest_pursuer = self._check_for_collisions()
        self.env_step += 1
        info = dict(collision_with_pursuer=collision_with_pursuer,
                    dist_btwn_nearest_pursuer=dist_btwn_nearest_pursuer,
                    env_step=self.env_step,
                    find_evader_num=len(self.obs_dict['pursuers_observing_list']))
        if self.env_step >= self._max_length:
            self.out_of_time = True 
            metrix = self.compute_metrix()
            info.update(metrix=metrix)
        done = self.out_of_time
        return self.obs_dict, done, info 

    def _check_for_collisions(self):

        collision_with_pursuer = [False for _ in self._pursuers]
        dist_btwn_nearest_pursuer = [np.inf for _ in self._pursuers]
        agent_inds = list(range(len(self._pursuers)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            dist_btwn = l2norm(self._pursuers[i].pos, self._pursuers[j].pos)
            combined_radius = self._pursuers[i].radius + self._pursuers[j].radius
            dist_btwn_nearest_pursuer[i] = min(dist_btwn_nearest_pursuer[i], dist_btwn - combined_radius)
            dist_btwn_nearest_pursuer[j] = min(dist_btwn_nearest_pursuer[j], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_pursuer[i] = True
                collision_with_pursuer[j] = True
                self.collision_pursuer_record[i][self.env_step] = 1
                self.collision_pursuer_record[j][self.env_step] = 1
        return collision_with_pursuer, dist_btwn_nearest_pursuer 
    
    def compute_metrix(self):
        """
        计算性能指标
        """ 
        tracking_ratio = {}
        collision_ratio = {}
        for key, value in self.observed_evader_record.items():
            tracking_ratio[key] = np.mean(value)
        for i in range(len(self.pursuers)):
            collision_ratio[i] = np.mean(self.collision_pursuer_record[i])
        if len(tracking_ratio) == 0:
             average_tracking_ratio = 0
        else:
            average_tracking_ratio = sum(tracking_ratio.values()) / len(tracking_ratio)
        average_collision_ratio = np.mean([collision_ratio[i] for i in range(len(self.pursuers))])
        deviation = np.mean([np.square(tracking_ratio[i] - average_tracking_ratio) for i in list(tracking_ratio.keys())])
        standard_deviation = np.sqrt(deviation)
        average_certainty_ratio = np.mean(self.observed_env_record) / self.env_step
        eval_metrics_dict = {
            'average_tracking_ratio': average_tracking_ratio,
            'standard_deviation': standard_deviation,
            'average_certainty_ratio': average_certainty_ratio,
            'average_collision_ratio': average_collision_ratio
        }
        return edict(eval_metrics_dict)


