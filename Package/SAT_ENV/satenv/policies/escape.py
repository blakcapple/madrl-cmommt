import numpy as np
from satenv.policies.internalPolicy import InternalPolicy
from satenv.env_utils.utils import wrap, l2norm


class Escape(InternalPolicy):
    """
    Modified Replusive Policy 
    """
    def __init__(self):
        InternalPolicy.__init__(self, str="Escape")

        self.now_goal = None
        

    def find_next_action(self, info, agent):
        agents = info['agents']
        evader_observe_pursuer_dict = info['evader_observe_pursuer_dict']
        obs_ag_num = len(evader_observe_pursuer_dict[agent.id])
        vec_to_enemy  = np.zeros(2)
        vec_to_wall = np.zeros(2)
        
        # 计算相对于视野内的敌人的排斥力
        for enemy_id in evader_observe_pursuer_dict[agent.id]:
            enemy = agents[enemy_id]
            enemy_pos = np.array([enemy.pos[0], enemy.pos[1]])
            repulsive_v = agent.pos - enemy_pos
            repulsive_v = (agent.pos - enemy_pos)/ np.sum(repulsive_v**2) if np.sum(repulsive_v**2) != 0 else [0,0]
            vec_to_enemy += repulsive_v

        # # 计算相对于边界的排斥力
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
        # vec_to_wall = vec_to_wall / np.sum(vec_to_wall**2) if np.sum(vec_to_wall**2) != 0 else [0,0]
        
        if obs_ag_num:
            # breakpoint()
            # 视野内有敌人，则向着敌人的排斥方向前进，排斥力和距离的平方成反比
            if np.sum(vec_to_wall**2) < 1:
                repulsive_vector = vec_to_wall
            else:
                repulsive_vector =  vec_to_enemy
            heading_global_frame_exp = np.arctan2(repulsive_vector[1], repulsive_vector[0])
            delta_heading = wrap(heading_global_frame_exp - agent.heading)
        else:
            # 视野内没有敌人，则向着随机产生的地图上的目标点前进
            if self.now_goal is None or (l2norm(agent.pos, self.now_goal) <= agent.near_goal_threshold):
                # 设置的目标点不靠近边界
                goal_x = (agent.max_x - agent.min_x)*(0.8) * np.random.rand() + agent.min_x
                goal_y = (agent.max_y - agent.min_y)*(0.8) * np.random.rand() + agent.min_y
                self.now_goal = np.array([goal_x, goal_y])

            vec_to_next_pose = self.now_goal - agent.pos
            heading_global_frame_exp = np.arctan2(vec_to_next_pose[1], vec_to_next_pose[0])
            delta_heading = wrap(heading_global_frame_exp - agent.heading)

        # if agent.dynamics_model.action_type == "R_THETA":
        #     if delta_heading > agent.max_heading_change: delta_heading = agent.max_heading_change
        #     if delta_heading < agent.min_heading_change: delta_heading = agent.min_heading_change

        select_speed = agent.pref_speed
        # for enemy_id in evader_observe_pursuer_dict[agent.id]:
        #     # Evader can not to close to the pursuer:
        #     enemy = agents[enemy_id]
        #     enemy_pos = np.array([enemy.pos[0], enemy.pos[1]])
        #     selected_heading = wrap(delta_heading + agent.heading)
        #     dx = select_speed * np.cos(selected_heading) * agent.dt_nominal
        #     dy = select_speed * np.sin(selected_heading) * agent.dt_nominal
        #     next_step_pos = agent.pos + np.array([dx, dy])
        #     dis_to_enemy = np.linalg.norm(next_step_pos - enemy_pos)
            # if dis_to_enemy < agent.capture_radius:
            #     select_speed = 0
        action = [select_speed, delta_heading]
        return action
