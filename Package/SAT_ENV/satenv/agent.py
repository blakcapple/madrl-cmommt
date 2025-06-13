import numpy as np
from satenv.configs.base_cfg import BaseConfig
import math 

class Agent(object):
    """ A disc-shaped object that has a policy, dynamics, sensors, and can move through the environment
    """

    def __init__(self, config: BaseConfig, start_pos, radius, sec_radius, 
                 comm_radius, pref_speed, max_heading_change, initial_heading,
                 policy, dynamics_model, id, group=0, render=False):
       
        self.config = config 
        self.policy = policy()
        self.dynamics_model = dynamics_model(self)
        self.render = render
        self.num_actions_to_store = 2
        self.action_dim = 2
        self.id = id
        self.group = group
        self.find_evader_num = 0
        self.dist_to_goal = 0.0
        self.near_goal_threshold = config.NEAR_GOAL_THRESHOLD
        self.dt_nominal = config.dt
        self.capture_radius = config.CAPTURE_RADIUS
        range = config.env_range
        self.min_x = range[0][0]
        self.max_x = range[0][1]
        self.min_y = range[1][0]
        self.max_y = range[1][1]
        self.max_heading_change = max_heading_change
        self.min_heading_change = -max_heading_change
        self.history_info = {key:[] for key in self.config.animation_colums}
        self.history_pos = []
        self.reset(pos=start_pos,
                   pref_speed=pref_speed,
                   radius=radius,
                   sec_radius=sec_radius,
                   heading=initial_heading,
                   comm_radius=comm_radius)

    def reset(self, pos, pref_speed, radius, sec_radius, heading, comm_radius):
        
        self.history_pos = []
        self.pos = np.array(pos, dtype='float64')
        self.heading = heading
        self.past_actions = np.zeros((self.num_actions_to_store, self.action_dim))
        self.radius = radius
        self.sec_radius = sec_radius
        self.pref_speed = pref_speed
        self.comm_radius = comm_radius
        self.vel = [pref_speed*math.cos(self.heading),pref_speed*math.cos(self.heading)] 
        self.t = 0.0
        self.step_num = 0
        self.is_done = False
        self.history_pos.append(self.pos)

    def __deepcopy__(self):
        """ Copy every attribute about the agent except its policy (since that may contain MBs of DNN weights) """
        cls = self.__class__
        obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k != 'policy':
                setattr(obj, k, v)
        return obj

    def find_next_action(self, dict_comm):

        if self.policy.type == "internal":
            action = self.policy.find_next_action(dict_comm, self)
        else:
            raise NotImplementedError

        return action

    def take_action(self, action, dt):

        # Store past actions
        self.past_actions = np.roll(self.past_actions, 1, axis=0)
        self.past_actions[0, :] = action

        self.dynamics_model.step(action, dt)
        self.history_pos.append(self.pos)
        self.to_vector()

        # Update time left so agent does not run around forever
        self.t += dt
        self.step_num += 1

    def print_agent_info(self):
        """ Print out a summary of the agent's current state. """
        print('----------')
        print('Global Frame:')
        print('(px,py):', self.pos)
        print('speed:', self.pref_speed)
        print('heading:', self.heading)
        print('Body Frame:')
        print('----------')

    def to_vector(self):
        
        global_state_dict = {
            't': self.t,
            'radius': self.radius,
            'pref_speed': self.pref_speed,
            'speed_global_frame': self.pref_speed,
            'pos_x': self.pos[0],
            'pos_y': self.pos[1],
            'alpha': self.heading,
        }
        global_state = np.array([val for val in global_state_dict.values()])

        if self.render:
            for key in self.config.animation_colums:
                self.history_info[key].append(global_state_dict[key])
        return global_state
