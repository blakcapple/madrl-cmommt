from satenv.dynamics.UnicycleDynamics import UnicycleDynamics
import numpy as np 
from satenv.policies import policy_dict
from satenv.agent import Agent

def get_pursuer_start_point(num, env_config):
    
    '''
    从环境的中心区域随机取点[[0.4, 0.6],[0.4, 0.6]]
    '''
    
    start_point = []
    min_x = env_config.ENV_RANGE[0][0]
    max_x = env_config.ENV_RANGE[0][1]
    min_y = env_config.ENV_RANGE[1][0]
    max_y = env_config.ENV_RANGE[1][1]
    x_len = max_x - min_x
    y_len = max_y - min_y
    for i in range(num):
        y = min_y + y_len * (np.random.rand()*0.2 + 0.4) 
        x = min_x + x_len * (np.random.rand()*0.2 + 0.4)
        start_pos = np.array([x, y])
        start_point.append(start_pos)
    return start_point

def get_evader_start_point(num, env_config):
    '''
    从边缘区域随机取点（不与追逐者的初始区域重叠）
    '''
    start_point = []
    min_x = env_config.ENV_RANGE[0][0]
    max_x = env_config.ENV_RANGE[0][1]
    min_y = env_config.ENV_RANGE[1][0]
    max_y = env_config.ENV_RANGE[1][1]
    x_len = max_x - min_x
    y_len = max_y - min_y
    for _ in range(num):
        p = np.random.rand()
        if p < 0.25:
            y = min_y + y_len * (np.random.rand()) 
            x = min_x + x_len * (np.random.rand()*0.3)
        elif p < 0.5:
            y = min_y + y_len * (np.random.rand()) 
            x = min_x + x_len * (np.random.rand()*0.3+0.7)
        elif p < 0.75:
            x = min_x + x_len * (np.random.rand())
            y = min_y + y_len * (np.random.rand()*0.3)
        elif p < 1:
            x = min_x + x_len * (np.random.rand())
            y = min_y + y_len * (np.random.rand()*0.3+0.7)
        start_pos = np.array([x, y])
        start_point.append(start_pos)
    return start_point


def init_agent(env_config, render=False, phase='train'):
    
    """
    generate agents with random parameters
    """
    # pursuer
    pursuer_num = env_config.pursuer_num
    pursuer_radius = env_config.pursuer_radius
    sec_radius = env_config.search_radius
    comm_radius = env_config.comm_radius
    max_heading_change = env_config.dt * env_config.max_angular_v
    pursuer_speed = env_config.pursuer_speed 
    pursuer_policie_name = env_config.pursuer_policy
    pursuer_dynamics_model = UnicycleDynamics  
    start_pos_list = get_pursuer_start_point(pursuer_num, env_config)
    agents_pursuer = []
    for i in range(pursuer_num):
        start_pos = start_pos_list[i]
        agent_args = {
            "config":env_config,
            "start_pos": start_pos,
            "radius": pursuer_radius,
            "sec_radius": sec_radius,
            "comm_radius": comm_radius,
            "pref_speed": pursuer_speed,
            "min_speed": 0.5 * pursuer_speed,
            "max_heading_change": max_heading_change,
            "initial_heading": np.random.uniform(-np.pi, np.pi),
            "policy": policy_dict[pursuer_policie_name],
            "dynamics_model": pursuer_dynamics_model,
            "id": i,
            "render":render
        }
        agent = Agent(**agent_args)
        agents_pursuer.append(agent)
    # evader
    evader_num = env_config.evader_num
    evader_radius = env_config.evader_radius
    evader_pref_speed = env_config.evader_speed
    evader_sec_radius = env_config.evader_sec_radius 
    evader_policie_name = 'escape'
    evader_dynamics_model = UnicycleDynamics  
    agents_evader = []
    start_pos_list = get_evader_start_point(evader_num, env_config)
    for i in range(evader_num):
        initial_heading = np.random.uniform(-np.pi, np.pi)
        agent_args = {
            "config":env_config,
            "start_pos": start_pos_list[i],
            "radius": evader_radius,
            "sec_radius": evader_sec_radius,
            "comm_radius": None,
            "pref_speed": evader_pref_speed,
            "max_heading_change": max_heading_change,
            "initial_heading": initial_heading,
            "policy": policy_dict[evader_policie_name],
            "dynamics_model": evader_dynamics_model,
            "id": i + pursuer_num,
            "group": 1,
            "render":render
        }
        agent = Agent(**agent_args)
        agents_evader.append(agent)

    return agents_pursuer, agents_evader