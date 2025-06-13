from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm
import numpy as np 

class Random(InternalPolicy):

    def __init__(self, use_matrix):
        super().__init__(self)
        self.now_goal = None 
        self.use_matrix_exploration = use_matrix

    def find_next_action(self, info, agent):
        close_to_wall = False # 是否靠近边界

        dis_x = [agent.pos[0] - agent.min_x, agent.pos[0] - agent.max_x]
        if np.abs(dis_x[0]) < np.abs(dis_x[1]):
            vec_x = dis_x[0]
        else:
            vec_x = dis_x[1]
        dix_y = [agent.pos[1] - agent.min_y, agent.pos[1] - agent.max_y]
        if np.abs(dix_y[0]) < np.abs(dix_y[1]):
            vec_y = dix_y[0]
        else:
            vec_y = dix_y[1]
        if np.abs(vec_x) < np.abs(vec_y):
            vec_to_wall = np.array([vec_x,0])
        else:
            vec_to_wall = np.array([0, vec_y])

        if np.sum(vec_to_wall**2) < 1:
            close_to_wall = True 

        # 随机设定移动目标
        if self.now_goal is None or (l2norm(agent.pos, self.now_goal) <= 1.3):
            # 设置的目标点不靠近边界
            new_goal = self.now_goal
            if self.now_goal is None :
                goal_x = (agent.max_x - agent.min_x)*(0.8) * np.random.rand() + agent.min_x
                goal_y = (agent.max_y - agent.min_y)*(0.8) * np.random.rand() + agent.min_y
                self.now_goal = np.array([goal_x, goal_y])
            else:
                while l2norm(self.now_goal, new_goal) < 2.5: 
                    goal_x = (agent.max_x - agent.min_x)*(0.8) * np.random.rand() + agent.min_x
                    goal_y = (agent.max_y - agent.min_y)*(0.8) * np.random.rand() + agent.min_y
                    self.now_goal = np.array([goal_x, goal_y])

        if close_to_wall:
            motion_vector = vec_to_wall
        else: 
            motion_vector = self.now_goal - agent.pos
        heading_exp = np.arctan2(motion_vector[1], motion_vector[0])
        delta_heading = wrap(heading_exp - agent.heading)

        return delta_heading