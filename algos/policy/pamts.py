from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm
import numpy as np 
from copy import deepcopy

class PAMTS(InternalPolicy):
    '''
    '''

    def __init__(self, agent_id, search_radius, max_env_range, evader_num, pursuer_num):

        super().__init__(self)

        self.agent_id = agent_id
        self.search_radius = search_radius
        self.max_env_range = max_env_range
        self.local_matrix_shape = (10, 10)
        self.global_grid = 0.5
        self.local_grid = 0.5
        self.global_matrix_shape = [int(self.max_env_range//self.global_grid), int(self.max_env_range//self.global_grid)]
        self.lambda_value = 50
        self.d_n = 0.5
        self.observe_matrix = np.zeros(self.global_matrix_shape) # 记录每个栅格最近一次被访问的时间
        self.last_observe_matrix = np.zeros(self.global_matrix_shape) # 记录上一次的观测矩阵
        self.evader_num = evader_num
        self.pursuer_num = pursuer_num
        self.agent_pos = None
        self.heading = None
        self.explore_matrix = np.zeros(self.local_matrix_shape)
        self.follow_matrix = np.zeros(self.local_matrix_shape)

    def _update_observe_matrix(self, env_step):

        origin_x = int(self.agent_pos[0] // self.global_grid)
        origin_y = int(self.agent_pos[1] // self.global_grid)
        r =  int(self.search_radius // self.global_grid) + 2
        for i in range(origin_x-r, origin_x+r+1):
            for j in range(origin_y-r, origin_y+r+1):
                if i < 0 or i >= self.global_matrix_shape[0] or j < 0 or j >= self.global_matrix_shape[1]:
                    continue
                x = i * self.global_grid
                y = j * self.global_grid
                if (x - self.agent_pos[0])**2 + (y - self.agent_pos[1])**2 <= self.search_radius**2:
                    self.observe_matrix[i][j] = env_step
                    if j - 1 >= 0:
                        self.observe_matrix[i][j-1] = env_step
                    if i - 1 >= 0:
                        self.observe_matrix[i-1][j] = env_step
                    if i - 1 >= 0 and j - 1 >= 0:   
                        self.observe_matrix[i-1][j-1] = env_step

    def get_local_matrix(self, matrix_grid, matrix):
        '''
        获取机器人周围的局部观测时间矩阵
        '''
        local_matrix = np.ones(self.local_matrix_shape) * np.inf
        for i in range(self.local_matrix_shape[0]):
            for j in range(self.local_matrix_shape[1]):
                local_x = (i-self.local_matrix_shape[0]//2)*self.local_grid + self.agent_pos[0]
                local_y = (j-self.local_matrix_shape[1]//2)*self.local_grid + self.agent_pos[1]
                if local_x < 0 or local_x > self.max_env_range or local_y < 0 or local_y > self.max_env_range:
                    continue
                pos = np.clip([local_x, local_y], 0, self.max_env_range-0.01)
                global_x = int(pos[0] // matrix_grid)
                global_y = int(pos[1] // matrix_grid)
                local_matrix[i][j] = matrix[global_x][global_y]
        return local_matrix

    def cal_intention_weight(self, observe_target, observe_friend, friend_find_target_num):

        """
        计算explore和track的权重
        """
        average_track_num = self.evader_num / self.pursuer_num
        find_target_num = len(observe_target)
        if find_target_num >= average_track_num: 
            track_weight = 1
            explore_weight = 0
        if find_target_num < average_track_num and find_target_num >= average_track_num*0.75:
            track_weight = 0.75
            explore_weight = 0.25
        if find_target_num < average_track_num*0.75 and find_target_num >= average_track_num*0.5:
            track_weight = 0.5
            explore_weight = 0.5
        if find_target_num < average_track_num*0.5 and find_target_num >= average_track_num*0.25:
            track_weight = 0.25
            explore_weight = 0.75
        if find_target_num < average_track_num*0.25:
            track_weight = 0.2
            explore_weight = 0.8
        track_weight = track_weight * friend_find_target_num / ((len(observe_friend)+1)*average_track_num)
        explore_weight =explore_weight * (1-friend_find_target_num / ((len(observe_friend)+1)*average_track_num))
        return track_weight, explore_weight


    def cal_cell_weight(self, track_weight, explore_weight, observe_target_pos, intial=False):
        """
        计算机器人周围栅格的权重
        """
        delta_observe = self.observe_matrix - self.last_observe_matrix
        local_observe_matrix = self.get_local_matrix(self.global_grid, self.observe_matrix)
        local_last_observe_matrix = self.get_local_matrix(self.global_grid, self.last_observe_matrix)
        local_matrix = self.get_local_matrix(self.global_grid, delta_observe)
        for i in range(local_matrix.shape[0]):
            for j in range(local_matrix.shape[1]):
                if local_matrix[i][j] == np.inf:
                    self.explore_matrix[i][j] = 0
                elif intial or local_last_observe_matrix[i][j] == 0:
                    self.explore_matrix[i][j] += 1
                else:
                    self.explore_matrix[i][j] += np.min([local_matrix[i][j] / self.lambda_value, 1])
                for target_pos in observe_target_pos:
                    grid_pos = [(i-(local_matrix.shape[0]-1)/2)*self.local_grid, (j-(local_matrix.shape[1]-1)/2)*self.local_grid] + self.agent_pos
                    dis = l2norm(grid_pos, target_pos)
                    if dis < self.d_n:
                        self.follow_matrix[i][j] += 1
                    elif dis < self.search_radius:
                        self.follow_matrix[i][j] += (1 - (1-1/3)* (dis - self.d_n) / (self.search_radius - self.d_n))
                    else:
                        self.follow_matrix[i][j] -= 0.1
                        self.follow_matrix[i][j] = max(0, self.follow_matrix[i][j])
                if len(observe_target_pos) == 0:
                    self.follow_matrix[i][j] = 0
        # 归一化follow_matrix和explore_matrix
        self.follow_matrix = self.follow_matrix / np.max(self.follow_matrix) if np.max(self.follow_matrix) != 0 else self.follow_matrix
        self.explore_matrix = self.explore_matrix / np.max(self.explore_matrix) if np.max(self.explore_matrix) != 0 else self.follow_matrix
        cell_weight_matrix = 1 * self.follow_matrix + explore_weight * self.explore_matrix
        # if len(observe_target_pos)>0:
        #     breakpoint()
        # print(observe_target_pos)
        return cell_weight_matrix

    def find_next_action(self, info):

        self.observe_matrix = info['observe_matrix']
        observe_target = info['observe_target']
        observe_friend = info['observe_friend']
        observe_target_pos = info['observe_target_pos']
        friend_find_target_num = info['friend_find_target_num']
        env_step = info['env_step']
        track_weight, explore_weight = self.cal_intention_weight(observe_target, observe_friend, friend_find_target_num)
        cell_weight_matrix = self.cal_cell_weight(track_weight, explore_weight, observe_target_pos, intial=env_step==0)
        # if env_step == 1000:
        #     breakpoint()
        # 计算矩阵中值最大的索引位置
        max_index = np.array(np.unravel_index(np.argmax(cell_weight_matrix), cell_weight_matrix.shape))
        motion_vector = max_index - np.array([(self.local_matrix_shape[0] - 1)//2, (self.local_matrix_shape[1]-1)//2])
        heading_exp = np.arctan2(motion_vector[1], motion_vector[0])
        delta_heading = wrap(heading_exp - self.heading)
        self.last_observe_matrix = deepcopy(self.observe_matrix)
        # print(motion_vector)
        return delta_heading   
    