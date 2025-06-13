import numpy as np
from collections import deque


class MutiAgentEpisodeBuffer:
    """
    多智能体buffer
    """
    def __init__(self, capacity, length, batch_size, n_robot, state_shape, action_shape, share_reward=False):
        # init 
        self.capacity = capacity
        self.length = length
        self.n_robot = n_robot
        self.batch_size = batch_size
        local_shape = state_shape['local_shape']
        matrix_shape = state_shape['matrix_shape']
        global_shape = state_shape['global_shape']
        mask_shape = state_shape['mask_shape']
        self.state_buf = {}
        self.state_buf['local_info'] = np.empty((capacity, length, n_robot, *local_shape), dtype=np.float32)
        self.state_buf['matrix_info'] = np.empty((capacity, length, 1, *matrix_shape), dtype=np.float32)
        self.state_buf['global_info'] = np.empty((capacity, length, global_shape), dtype=np.float32)
        self.state_buf['mask_info'] = np.empty((capacity, length, n_robot, mask_shape), dtype=np.float32)
        self.state_buf['id_info'] = np.empty((capacity, length, n_robot, n_robot), dtype=np.float32)
        self.act_buf = np.empty((capacity, length, n_robot, action_shape), dtype=np.float32)
        if share_reward:
            self.rew_buf = np.empty((capacity, length, 1), dtype=np.float32) # share reward
        else:
            self.rew_buf = np.empty((capacity, length, n_robot), dtype=np.float32)
        self.done_buf = np.empty((capacity, length, 1), dtype=np.float32)
        self.ep_cntr = 0
        self.step_cntr = 0 ## 记录step数

        self.gamma = 0.99

    def push(self, state, action=None, reward=None, done=None):

        #store
        ep_idx = self.ep_cntr % self.capacity
        self.state_buf['local_info'][ep_idx][self.step_cntr] = state['local_info']
        self.state_buf['matrix_info'][ep_idx][self.step_cntr] = state['matrix_info']
        self.state_buf['global_info'][ep_idx][self.step_cntr] = state['global_info']
        self.state_buf['mask_info'][ep_idx][self.step_cntr] = state['mask_info']
        self.state_buf['id_info'][ep_idx][self.step_cntr] = state['id_info']
        if action is not None:
            self.act_buf[ep_idx][self.step_cntr] = action
            self.rew_buf[ep_idx][self.step_cntr] = reward
            self.done_buf[ep_idx][self.step_cntr] = done
        self.step_cntr +=1
        if self.step_cntr % self.length == 0 :
            self.step_cntr = 0
            self.ep_cntr +=1
    
    def sample_batch(self):
        size = min(self.ep_cntr, self.capacity)
        idxs = np.random.randint(0, size, size=self.batch_size)
        local_info = self.state_buf['local_info'][idxs]
        matrix_info = self.state_buf['matrix_info'][idxs]
        global_info = self.state_buf['global_info'][idxs]
        mask_info = self.state_buf['mask_info'][idxs]
        id_info = self.state_buf['id_info'][idxs]
        state = dict(local_info=local_info, matrix_info=matrix_info, global_info=global_info, mask_info=mask_info, id_info=id_info)
        batch = dict(state=state,
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     idxs=idxs)
        return batch

    def ready(self):
        if self.ep_cntr>=self.batch_size:
            return True
    
    def __len__(self):
        return min(self.ep_cntr, self.capacity)
