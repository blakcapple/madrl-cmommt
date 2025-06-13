from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm
import numpy as np 

# constant parameters
d1 = 0.5
d2 = 0.8
sense_range = 1.25
predictive_tracking_range = 2
dr1 = 1.25
dr2 = 2

def target_force_func(distance):
    if distance <= d1:
        force = -1 + distance*(1/d1)
    elif distance <= d2:
        force = 0 + (distance - d1)*(1/(d2-d1))
    elif distance <= sense_range:
        force = 1 
    elif distance <= predictive_tracking_range:
        force = 1 - (distance - sense_range)*(1/(predictive_tracking_range-sense_range))

    else: force = 0

    return force 

def robot_force_func(distance):

    if distance <= dr1:
        force = -1 
    elif distance <= dr2:
        force = -1 + (distance - dr1)*(1/(dr2-dr1))
    
    else: force = 0

    return force 


class ACMOMMT(InternalPolicy):
    '''
    基于ACMOMMT算法, 在吸引向量和排斥向量的基础上加入基于概率矩阵的探索引导，提高算法对环境的探索能力
    '''

    def __init__(self, use_matrix):
        super().__init__(self)
        self.now_goal = None 
        self.use_matrix_exploration = use_matrix

    def find_next_action(self, info, agent):
        close_to_wall = False # 是否靠近边界
        random_move = False # 是否采用随机移动策略
        id = agent.id
        agents = info['agents']
        target_list = info['pursuer_observe_evader_dict'][id]
        robot_list =  info['friend_observe_dict'][id]
        local_matrix = info['local_matrix']
        agent_pos = agent.pos
        explore_vector = np.array([0., 0.])
        if self.use_matrix_exploration:
            # 基于概率举矩阵计算质心，质心相对于自身的坐标作为探索方向指引
            centroid = np.array([0., 0.])
            total_weight = 0
            local_matrix = np.mean(local_matrix, axis=0)
            for i in range(local_matrix.shape[0]):
                for j in range(local_matrix.shape[1]):
                    weight = 1 / (np.abs(local_matrix[i,j] - 0.5) / 0.5 + 1)
                    centroid[0] += weight * i
                    centroid[1] += weight * j
                    total_weight += weight

            if total_weight != 0:
                centroid /= total_weight
            explore_vector = centroid - np.array([(local_matrix.shape[0]-1) / 2 , (local_matrix.shape[1]-1) / 2])

        # 计算智能体相对于目标产生的引力
        target_vector = np.array([0., 0.])
        for target_id in target_list:
            weight = 1 # 对于重复观测的目标对象会给予一个小权重
            target = agents[target_id]
            for friend_id in robot_list:
                if l2norm(target.pos, agents[friend_id].pos) < sense_range:
                    weight -= 0.2
            weight = max(weight, 0)
            vector = target.pos - agent_pos
            distance = np.linalg.norm(vector) 
            norm_vector = vector / distance if distance != 0 else vector
            force = target_force_func(distance)
            target_vector += force * norm_vector * weight

        # 计算智能体相对于其他智能体的斥力
        robot_vector = np.array([0.,0.])
        for robot_id in robot_list:
            robot = agents[robot_id]
            vector = robot.pos - agent_pos
            distance = np.linalg.norm(vector)
            norm_vector = vector / distance if distance != 0 else vector
            force = robot_force_func(distance)
            robot_vector += force * norm_vector

        # 归一化向量
        robot_vector = robot_vector / np.linalg.norm(robot_vector) if np.linalg.norm(robot_vector) != 0 else robot_vector
        target_vector = target_vector / np.linalg.norm(target_vector) if np.linalg.norm(target_vector) != 0 else target_vector
        explore_vector = explore_vector / np.linalg.norm(explore_vector) if np.linalg.norm(explore_vector) != 0 else explore_vector

        # 加权求和
        motion_vector = robot_vector + 2*target_vector + explore_vector

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
        if np.sum(motion_vector**2) == 0:
            random_move = True 

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
        elif random_move: 
            motion_vector = self.now_goal - agent.pos
        heading_exp = np.arctan2(motion_vector[1], motion_vector[0])
        delta_heading = wrap(heading_exp - agent.heading)

        return delta_heading