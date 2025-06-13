import numpy as np
from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm

class RandomMove(InternalPolicy):
    """
    Random Move Policy: constantly Move to a random set goal 
    """
    def __init__(self):
        InternalPolicy.__init__(self, str="Escape")

        self.now_goal = None
        
    def find_next_action(self, info, agent):

        if self.now_goal is None or (l2norm(agent.pos, self.now_goal) <= agent.near_goal_threshold):
            # 设置的目标点不靠近边界
            goal_x = (agent.max_x - agent.min_x)*(0.8) * np.random.rand() + agent.min_x
            goal_y = (agent.max_y - agent.min_y)*(0.8) * np.random.rand() + agent.min_y
            self.now_goal = np.array([goal_x, goal_y])

        vec_to_next_pose = self.now_goal - agent.pos
        heading_global_frame_exp = np.arctan2(vec_to_next_pose[1], vec_to_next_pose[0])
        delta_heading = wrap(heading_global_frame_exp - agent.heading)

        select_speed = agent.pref_speed

        action = [select_speed, delta_heading]
        return action