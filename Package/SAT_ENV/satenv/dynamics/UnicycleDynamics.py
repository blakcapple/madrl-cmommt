import numpy as np
from satenv.env_utils.utils import wrap

class UnicycleDynamics(object):
    """ Convert a speed & heading to a new state according to Unicycle Kinematics model.

    """

    def __init__(self, agent):
        self.agent = agent
        self.action_type = "R_THETA"

    def step(self, action, dt):

        selected_speed = action[0]
        delta_heading = action[1] * dt
        if delta_heading > self.agent.max_heading_change: 
            delta_heading = self.agent.max_heading_change
        if delta_heading < self.agent.min_heading_change: 
            delta_heading = self.agent.min_heading_change
        selected_heading = wrap(delta_heading + self.agent.heading)
        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.agent.pos += np.array([dx, dy])

        cmb_array = np.concatenate(
            [np.array([[self.agent.max_x, self.agent.max_y]]), self.agent.pos[np.newaxis, :]], axis=0)
        self.agent.pos = np.min(cmb_array, axis=0)
        cmb_array = np.concatenate(
            [np.array([[self.agent.min_x, self.agent.min_y]]), self.agent.pos[np.newaxis, :]], axis=0)
        self.agent.pos = np.max(cmb_array, axis=0)

        self.agent.vel[0] = selected_speed * np.cos(selected_heading)
        self.agent.vel[1] = selected_speed * np.sin(selected_heading)
        self.agent.pref_speed = selected_speed
        self.agent.heading = selected_heading
