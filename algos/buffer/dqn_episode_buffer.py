import numpy as np
from collections import deque


class MutiAgentEpisodeBuffer:
    """
    多智能体buffer
    """
    def __init__(self, capacity, length, batch_size, observation_space, n_robot, action_shape, share_reward=False):
        # init 
        self.capacity = capacity
        self.length = length
        self.n_robot = n_robot
        self.batch_size = batch_size
        self.state_buf = {}
        for key, val in observation_space.items():
            self.state_buf[key] = np.zeros((capacity, length+1, *val.shape), dtype=val.dtype.name)
        self.act_buf = np.empty((capacity, length, n_robot, action_shape), dtype=np.float32)
        if share_reward:
            self.rew_buf = np.empty((capacity, length, 1), dtype=np.float32) # share reward
        else:
            self.rew_buf = np.empty((capacity, length, n_robot), dtype=np.float32)
        self.done_buf = np.empty((capacity, length+1, 1), dtype=np.float32)
        self.ep_cntr = 0
        self.step_cntr = 0 ## 记录step数
        self.gamma = 0.99

    def push(self, state, action=None, reward=None, done=None):

        #store
        ep_idx = self.ep_cntr % self.capacity
        for key in self.state_buf.keys():
            assert key in state.keys(), print(state.keys())
            self.state_buf[key][ep_idx][self.step_cntr+1] = state[key].copy()
        if action is not None:
            self.act_buf[ep_idx][self.step_cntr] = action
            self.rew_buf[ep_idx][self.step_cntr] = reward
            self.done_buf[ep_idx][self.step_cntr+1] = done
        self.step_cntr +=1
        if self.step_cntr % self.length == 0 :
            self.step_cntr = 0
            self.ep_cntr +=1
    
    def sample_batch(self):
        size = min(self.ep_cntr, self.capacity)
        idxs = np.random.randint(0, size, size=self.batch_size)
        state = {}
        for key in self.state_buf.keys():
            state[key] = self.state_buf[key][idxs].copy()
        batch = dict(state=state,
                    act=self.act_buf[idxs].copy(),
                    rew=self.rew_buf[idxs].copy(),
                    done=self.done_buf[idxs].copy(),
                    idxs=idxs)
        return batch

    def ready(self):
        if self.ep_cntr>=self.batch_size:
            return True
    
    def __len__(self):
        return min(self.ep_cntr, self.capacity)
