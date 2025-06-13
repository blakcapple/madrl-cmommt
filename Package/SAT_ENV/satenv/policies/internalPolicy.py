import numpy as np
from satenv.env_utils.utils import wrap


class InternalPolicy(object):
    """
    Convert an observation to an action completely within the environment (for model-based/pre-trained, simulated agents).
    Please see the possible subclasses at :ref:`all_internal_policies`.
    """
    def __init__(self, str="Internal"):
        self.str = str
        self.type = "internal"
        self.is_still_learning = False
        self.is_external = False

    def near_goal_smoother(self, dist_to_goal, pref_speed, heading, raw_action):
        """ Linearly ramp down speed/turning if agent is near goal, stop if close enough.

        I think this is just here for convenience, but nobody uses it? We used it on the jackal for sure.
        """
        kp_v = 0.5
        kp_r = 1

        if dist_to_goal < 2.0:
            near_goal_action = np.empty((2,1))
            pref_speed = max(min(kp_v * (dist_to_goal-0.1), pref_speed), 0.0)
            near_goal_action[0] = min(raw_action[0], pref_speed)
            turn_amount = max(min(kp_r * (dist_to_goal-0.1), 1.0), 0.0) * raw_action[1]
            near_goal_action[1] = wrap(turn_amount + heading)
        if dist_to_goal < 0.3:
            near_goal_action = np.array([0., 0.])
        else:
            near_goal_action = raw_action

        return near_goal_action

    def find_next_action(self, obs, info, agent):
        """ Use the provided inputs to select a commanded action [heading delta, speed]
            To be implemented by children.
        """
        raise NotImplementedError