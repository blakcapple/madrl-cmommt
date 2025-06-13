from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm
import numpy as np 

# constant parameters
d1 = 0.5
d2 = 0.8
d3 = 1.25
predictive_tracking_range = 2
dr1 = 1.25
dr2 = 2

def target_force_func(distance):
    if distance <= d1:
        force = -1 + distance*(1/d1)
    elif distance <= d2:
        force = 0 + (distance - d1)*(1/(d2-d1))
    elif distance <= d3:
        force = 1 
    elif distance <= predictive_tracking_range:
        force = 1 - (distance - d3)*(1/(predictive_tracking_range-d3))

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

    def __init__(self):
        super().__init__(self)
        self.now_goal = None 

    def find_next_action(self, info, agent):
        close_to_wall = False # 是否靠近边界
        random_move = False # 是否采用随机策略
        id = agent.id
        agents = info['agents']
        target_list = info['pursuer_observe_evader_dict'][id]
        robot_list =  info['friend_observe_dict'][id]
        agent_pos = agent.pos_global_frame
        target_vector = np.array([0., 0.])
        # 计算智能体相对于目标产生的引力
        for target_id in target_list:
            target = agents[target_id]
            vector = target.pos_global_frame - agent_pos
            distance = np.linalg.norm(vector)
            norm_vector = vector / distance # 单位向量
            force = target_force_func(distance)
            target_vector += force * norm_vector
        # 计算智能体相对于其他智能体的斥力
        robot_vector = np.array([0.,0.])
        for robot_id in robot_list:
            robot = agents[robot_id]
            vector = robot.pos_global_frame - agent_pos
            distance = np.linalg.norm(vector)
            norm_vector = vector / distance # 单位向量
            force = robot_force_func(distance)
            robot_vector += force * norm_vector
        
        # 计算相对于边界的排斥力
        dis_x = [agent.pos_global_frame[0] - agent.min_x, agent.pos_global_frame[0] - agent.max_x]
        if np.abs(dis_x[0]) < np.abs(dis_x[1]):
            vec_x = dis_x[0]
        else:
            vec_x = dis_x[1]
        dix_y = [agent.pos_global_frame[1] - agent.min_y, agent.pos_global_frame[1] - agent.max_y]
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
        if np.sum(robot_vector**2) == 0:
            random_move = True 

        # 随机设定移动目标
        if self.now_goal is None or (l2norm(agent.pos_global_frame, self.now_goal) <= 1.3):
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
            motion_vector = self.now_goal - agent.pos_global_frame
        else:
            motion_vector = target_vector + robot_vector
        heading_global_frame_exp = np.arctan2(motion_vector[1], motion_vector[0])
        delta_heading = wrap(heading_global_frame_exp - agent.heading_global_frame)

        # if delta_heading > agent.max_heading_change: delta_heading = agent.max_heading_change
        # if delta_heading < agent.min_heading_change: delta_heading = agent.min_heading_change

        select_speed = agent.pref_speed
        action = [select_speed, delta_heading]
        return action