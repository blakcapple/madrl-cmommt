from gym import spaces
from env_wrapper.popenv import POPEnv
import numpy as np
from copy import deepcopy
import torch
from .env_utils.utils import l2norm

class MetaEnv(POPEnv):
    
    def __init__(self, env, config, use_model=True):
        
        super().__init__(env, config)
        
        self.observation_space = {
            'friend_info':spaces.Box(low=0, high=1, shape=[self.agent_num, self.agent_num-1, self.agent_info_shape]),
            'target_info':spaces.Box(low=0, high=1, shape=(self.agent_num, self.max_evader_in_obs, self.agent_info_shape)),
            'matrix_info':spaces.Box(low=0, high=1, shape=(self.agent_num, *self.local_matrix_shape,)),
            'agent_info':spaces.Box(low=0, high=1, shape=(self.agent_num, self.agent_info_shape,)),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.agent_num, self.max_evader_in_obs+2,), dtype=bool),
            'target_mask': spaces.Box(low=0, high=1, shape=(self.agent_num, self.max_evader_in_obs,)),
            'friend_mask': spaces.Box(low=0, high=1, shape=(self.agent_num, self.agent_num-1,)),                         
        }
        if self.use_global_info:
            if self.use_agent_specific_global_info:
                self.observation_space['global_info'] = spaces.Box(low=0, high=1, shape=(self.agent_num, 
                                                        (self.max_pursuer_in_obs+self.max_evader_in_obs+1)*self.agent_info_shape,))
            else:
                self.observation_space['global_info'] = spaces.Box(low=0, high=1, shape=(1, 
                                                        (self.max_pursuer_in_obs+self.max_evader_in_obs+1)*self.agent_info_shape,))
        # 视野内目标的最大数量 
        self.action_space = spaces.Discrete(self.max_evader_in_obs+2)# 0 探索策略 1 无目标跟踪策略 2……n+1 目标索引
        # self.action_space = spaces.Discrete(2)
        self.share_observation_space = deepcopy(self.observation_space)
        self.action_repeat = 1
        self.use_model = use_model
        self.exploration_step = 0
        self.use_explore_model = config.use_explore_model
    
    def set_model(self, tracking_policy, exploring_policy):

        self._set_tacking_model(tracking_policy)
        self._set_exploring_model(exploring_policy)
        
    def _set_tacking_model(self, model):
        """
        加载跟踪网络
        """
        self.tracking_model = model
        print('load tracking policy !')
        
    def _set_exploring_model(self, model):
        
        """
        加载探索网络
        """
        self.exploring_model = model
        print('load exploring policy !')

    def get_local_vector_obs(self, obs_dict):
        
        """
        得到观测列表 (包含所有智能体观测到的目标信息)
        """
        friend_observe_dict = obs_dict['friend_observe_dict']
        pursuer_observe_evader_dict = obs_dict['pursuer_observe_evader_dict']
        agent_vector = np.zeros((self.agent_num, self.agent_info_shape))
        all_friend_vector = np.zeros((self.agent_num, self.max_pursuer_in_obs, self.agent_info_shape))
        all_target_vector = np.zeros((self.agent_num, self.max_evader_in_obs, self.agent_info_shape))
        action_mask = np.zeros((self.agent_num, self.max_evader_in_obs+2)) # 动作屏蔽， 1 表示非法动作 0 表示合法动作
        # action_mask = np.zeros((self.agent_num, 2))
        target_mask = np.zeros((self.agent_num, self.max_evader_in_obs)) # 屏蔽无效的目标信息 0 表示无效
        friend_mask = np.zeros((self.agent_num, self.max_pursuer_in_obs)) # 屏蔽无效的队友信息
        agent_mask = np.zeros((self.agent_num, 1))
        for index in range(self.agent_num):
            if index >= len(self.env_core.pursuers):
                break
            agent_mask[index] = 1 # 置有效位
            agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
            agent_vector[index, :2] = agent_pos
            agent_vector[index, 2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
            agent_vector[index, 3] = self.env_core.pursuers[index].heading / (np.pi*2)

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
            action_mask[index, min(self.max_evader_in_obs, len(target_info))+2:] = 1
            if len(target_info) >= 1:
                action_mask[index, 1] = 1

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
                    agent_vector[index, :2] = agent_pos
                    agent_vector[index, 2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
                    agent_vector[index, 3] = self.env_core.pursuers[index].heading / (np.pi*2)
                    
                    friend_info = [i for i in range(len(self.env_core.pursuers))]
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
                all_friend_vector = np.zeros((self.max_pursuer_in_obs+1, self.agent_info_shape))
                all_target_vector = np.zeros((self.max_evader_in_obs, self.agent_info_shape))
                for i in range(len(self.env_core.pursuers)):
                    if i >= self.max_pursuer_in_obs:
                        break
                    agent_pos = self.env_core.pursuers[index].pos / self.max_env_range
                    all_friend_vector[i][:2] = agent_pos
                    all_friend_vector[i][2] = self.env_core.pursuers[index].pref_speed if not self._ignore_speed else 1
                    all_friend_vector[i][3] = self.env_core.pursuers[index].heading / (np.pi*2)
                for i in range(len(self.env_core.evaders)):
                    if i >= self.max_evader_in_obs:
                        break
                    agent_pos = self.env_core.evaders[index].pos / self.max_env_range
                    all_target_vector[i][:2] = agent_pos
                    all_target_vector[i][2] = self.env_core.evaders[index].pref_speed if not self._ignore_speed else 1
                    all_target_vector[i][3] = self.env_core.evaders[index].heading / (np.pi*2)
                global_info = np.concatenate(
                                [all_friend_vector[None,:], all_target_vector[None,:]], axis=1).reshape(1,-1)
                observation['global_info'] = global_info.copy()

        return observation

    def trans_obs(self, obs_dict:dict):
        
        '''
        按照智能体id来排列obs
        '''
        obs = {i:dict() for i in range(self.agent_num)}
        for key, val in obs_dict.items():
            for id in obs.keys():
                obs[id][key] = val[id].unsqueeze(0).unsqueeze(0)
        return obs
    
    def cal_explore_vetor(self, agent_id):

        centroid = np.array([0., 0.])
        total_weight = 0
        local_matrix = np.mean(self.local_matrix[agent_id], axis=0)
        for i in range(local_matrix.shape[0]):
            for j in range(local_matrix.shape[1]):
                weight = 1 / (np.abs(local_matrix[i,j] - 0.5) / 0.5 + 1)
                centroid[0] += weight * i
                centroid[1] += weight * j
                total_weight += weight

        if total_weight != 0:
            centroid /= total_weight
        explore_vector = centroid - np.array([(local_matrix.shape[0]-1) / 2 , (local_matrix.shape[1]-1) / 2])

        return explore_vector
    def cal_robot_vector(self, agent_pos, robot_list):
        dr1 = 1.25
        dr2 = 2
        def robot_force_func(distance):
            if distance <= dr1:
                force = -1 
            elif distance <= dr2:
                force = -1 + (distance - dr1)*(1/(dr2-dr1))
            else: force = 0
            return force 
        robot_vector = np.array([0.,0.])
        for robot_id in robot_list:
            robot = self.env_core.agents[robot_id]
            vector = robot.pos - agent_pos
            distance = np.linalg.norm(vector)
            norm_vector = vector / distance if distance != 0 else vector
            force = robot_force_func(distance)
            robot_vector += force * norm_vector
        return robot_vector

    def cal_explore_action(self, obs):

        agent_id = obs['agent_id']
        # agent_pos = obs['agent_pos']
        agent_heading = obs['agent_heading']
        # 计算基于概率矩阵的探索向量
        explore_vector = self.cal_explore_vetor(agent_id)
        explore_vector = explore_vector / np.linalg.norm(explore_vector) if np.linalg.norm(explore_vector) != 0 else explore_vector
        # 计算机器人之间的互斥向量
        robot_vector = self.cal_robot_vector(self.env_core.pursuers[agent_id].pos, obs['friend_observe_dict'][agent_id])
        robot_vector = robot_vector / np.linalg.norm(robot_vector) if np.linalg.norm(robot_vector) != 0 else robot_vector
        motion_vector = robot_vector + explore_vector
        heading_exp = np.arctan2(motion_vector[1], motion_vector[0])
        from satenv.env_utils.utils import wrap, l2norm
        delta_heading = wrap(heading_exp - agent_heading)

        return delta_heading

    def meta_actions_transform(self, obs, obs_dict, actions):
        
        """meta policy action choose 
        """
        for key, val in obs.items():
            obs[key] = torch.tensor(val, dtype=torch.float32)
        obs = self.trans_obs(obs)
        low_actions = np.zeros((self.agent_num, 1), dtype=np.uint8)
        env_actions = []
        for i, a in enumerate(actions):
            if a == 0:
                if self.use_explore_model:
                    value = self.exploring_model(obs[i]).detach().numpy()
                    low_action = np.argmax(value)
                    env_action = self.actions.map[low_action]
                else:
                    ob = {}
                    ob['agent_id'] = i
                    ob['agent_heading'] = self.env_core.pursuers[i].heading
                    ob['friend_observe_dict'] = obs_dict['friend_observe_dict']
                    env_action = self.cal_explore_action(ob)
                env_actions.append(env_action)
            else:
                # only consider the target that agent select
                if a >= 2:
                    mask = torch.zeros(self.max_evader_in_obs, dtype=torch.float32)
                    mask[a-2] = 1
                    obs[i]['target_mask'] = mask.unsqueeze(0).unsqueeze(0)
                value = self.tracking_model(obs[i]).detach().numpy()
                low_action = np.argmax(value)
                env_action = self.actions.map[low_action]
                env_actions.append(env_action)
        # low_actions = low_actions.reshape(-1)
        # env_actions = self._actions_transform(low_actions)
        return env_actions
    
    def rule_actions_tranform(self, obs, actions, env_step):
        for key, val in obs.items():
            obs[key] = torch.tensor(val, dtype=torch.float32)
        obs = self.trans_obs(obs)
        low_actions = np.zeros((self.agent_num, 1), dtype=np.uint8)
        env_actions = []
        for i , a in enumerate(actions):
            target_info = obs[i]['target_info']
            target_num = torch.sum(target_info == 0) / self.agent_info_shape
            if env_step <= 30:
                a = 0
            else:
                if target_num > 0:
                    a = 2
                else:
                    a = 1
            if a == 0:
                if self.use_explore_model:
                    value = self.exploring_model(obs[i]).detach().numpy()
                    low_action = np.argmax(value)
                    env_action = self.actions.map[low_action]
                else:
                    ob = {}
                    ob['agent_id'] = i
                    ob['agent_heading'] = self.env_core.pursuers[i].heading
                    env_action = self.cal_explore_action(ob)
                env_actions.append(env_action)
            else:
                # only consider the target that agent select
                if a >= 2:
                    mask = torch.zeros(self.max_evader_in_obs, dtype=torch.float32)
                    mask[a-2] = 1
                    obs[i]['target_mask'] = mask.unsqueeze(0).unsqueeze(0)
                value = self.tracking_model(obs[i]).detach().numpy()
                low_action = np.argmax(value)
                env_action = self.actions.map[low_action]
                env_actions.append(env_action)
            
        return env_actions
    
    def meta_actions_transform2(self, obs, obs_dict, actions):

        """meta policy action choose 
        """
        for key, val in obs.items():
            obs[key] = torch.tensor(val, dtype=torch.float32)
        obs = self.trans_obs(obs)
        env_actions = []
        for i, a in enumerate(actions):
            if a == 0:
                if self.use_explore_model:
                    value = self.exploring_model(obs[i]).detach().numpy()
                    low_action = np.argmax(value)
                    env_action = self.actions.map[low_action]
                else:
                    ob = {}
                    ob['agent_id'] = i
                    ob['agent_heading'] = self.env_core.pursuers[i].heading
                    ob['friend_observe_dict'] = obs_dict['friend_observe_dict']
                    env_action = self.cal_explore_action(ob)
                env_actions.append(env_action)
            else:
                mask = torch.zeros(self.max_evader_in_obs, dtype=torch.float32)
                mask[0] = 1
                obs[i]['target_mask'] = mask.unsqueeze(0).unsqueeze(0)
                value = self.tracking_model(obs[i]).detach().numpy()
                low_action = np.argmax(value)
                env_action = self.actions.map[low_action]
                env_actions.append(env_action)

        return env_actions

    def adaptive_actions_transform(self, obs, actions):
        
        for key, val in obs.items():
            obs[key] = torch.tensor(val, dtype=torch.float32)
        obs = self.trans_obs(obs)
        low_actions = np.zeros((self.agent_num, 1), dtype=np.uint8)
        env_actions = []
        for i, a in enumerate(actions):
            target_info = obs[i]['target_info']
            target_num = torch.sum(target_info != 0) / self.agent_info_shape
            if target_num > 0:
                a = 2
            else:
                a = 0    
            if a == 0:
                if self.use_explore_model:
                    value = self.exploring_model(obs[i]).detach().numpy()
                    low_action = np.argmax(value)
                    env_action = self.actions.map[low_action]
                else:
                    ob = {}
                    ob['agent_id'] = i
                    ob['agent_heading'] = self.env_core.pursuers[i].heading
                    env_action = self.cal_explore_action(ob)
                env_actions.append(env_action)
            else:
                # only consider the target that agent select
                if a >= 2:
                    mask = torch.zeros(self.max_evader_in_obs, dtype=torch.float32)
                    mask[a-2] = 1
                    obs[i]['target_mask'] = mask.unsqueeze(0).unsqueeze(0)
                value = self.tracking_model(obs[i]).detach().numpy()
                low_action = np.argmax(value)
                env_action = self.actions.map[low_action]
                env_actions.append(env_action)

        return env_actions
        
    def step(self, actions:np.array):

        # 没有目标时搜索，有目标时跟踪

        for _ in range(self.action_repeat):
            if isinstance(self.action_space, spaces.Discrete):
                actions = np.array(actions).reshape(-1)
            actions = self.meta_actions_transform(self.pre_obs, self.pre_obs_dict, actions)

            for _ in range(self.duration_time):
                obs_dict, done, info = self.env_core.step(actions)

                if done:
                    break
            self._update_global_probability_matrix(obs_dict)
            if self._use_matrix:
                self._get_local_matrix(obs_dict)
            if self._use_map_obs:
                obs = self.get_local_map_obs(obs_dict)
            else:
                obs = self.get_local_vector_obs(obs_dict)
            rewards, rewards_info = self.compute_rewards(obs_dict, info)
            if done:
                info = info['metrix']
            self.pre_obs = deepcopy(obs)
            self.pre_obs_dict = deepcopy(obs_dict)
        
        return obs, rewards, done, info, rewards_info