import numpy as np
from distributed_marl.algos.utils.segment_tree import MinSegmentTree, SumSegmentTree
import random
from collections import deque
from gym import spaces
import torch as T

class MultiAgentReplayBuffer:
    
    def __init__(self, capacity, batch_size, observation_space:spaces.Dict, n_robot, action_shape, n_step=1, share_reward=False):

        self.capacity = capacity
        self.n_robot = n_robot
        self.batch_size = batch_size 
        self.state_buf = {}
        self.state2_buf = {}
        for key, val in observation_space.items():
            self.state_buf[key] = np.empty((capacity, *val.shape), dtype=val.dtype.name)
            self.state2_buf[key] = np.empty((capacity, *val.shape), dtype=val.dtype.name)
        if share_reward:
            self.rew_buf = np.empty((capacity, 1), dtype=np.float32)
        else:
            self.rew_buf = np.empty((capacity, n_robot), dtype=np.float32)
        self.act_buf = np.empty((capacity, n_robot, action_shape), dtype=np.float32)
        self.done_buf = np.empty((capacity), dtype=np.float32)
        self.mem_cntr = 0
        self.n_step_buf = deque(maxlen=3)
        self.n_step = n_step 
        self.gamma = 0.99
            

    def push(self, state, action, reward, state2, done):
        
        transition = (state, action, reward, state2, done)
        self.n_step_buf.append(transition)
        if len(self.n_step_buf) < self.n_step:
            return () 
        
        # make a n-step transition
        reward, state2, done = self._get_n_step_info(self.n_step_buf, self.gamma)
        state, action = self.n_step_buf[0][:2]

        #store
        idx = self.mem_cntr % self.capacity
        for key in self.state_buf.keys():
            assert key in state.keys(), print(state.keys())
            self.state_buf[key][idx] = state[key].copy()
            self.state2_buf[key][idx] = state2[key].copy()


        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.done_buf[idx] = done
        self.mem_cntr +=1
        return self.n_step_buf[0]


    def sample_batch(self):
        size = min(self.mem_cntr, self.capacity)
        idxs = np.random.randint(0, size, size=self.batch_size)
        state = {}
        state2 = {}
        for key in self.state_buf.keys():
            state[key] = self.state_buf[key][idxs].copy()
            state2[key] = self.state2_buf[key][idxs].copy()
        batch = dict(state=state,
                     state2=state2,
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     idxs=idxs)
        return batch

    def ready(self):
        if self.mem_cntr>=self.batch_size:
            return True
    
    def __len__(self):
            return min(self.mem_cntr, self.capacity)

    def _get_n_step_info(self, n_step_buffer, gamma):

        # info of the last transition 
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d  = transition[-3:]
            rew = r + gamma * rew * (1-d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        
        return rew, next_obs, done 

class MultiPrioritizedReplayBuffer(MultiAgentReplayBuffer):
    """
    根据TD ERROR 的优先级来确定经验回放数组采样的优先级
    """
    def __init__(self, capacity, batch_size, n_robot, observation_space, action_shape, alpha, n_step=1, share_reward=False):
        assert alpha >= 0
        super(MultiPrioritizedReplayBuffer, self).__init__(capacity, batch_size, observation_space, n_robot, action_shape, n_step, share_reward)
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        self.alpha = alpha
        while tree_capacity < self.capacity:
            tree_capacity *=2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def push(self, state, action, reward, state2, done):
        transition = super().push(state, action, reward, state2, done)
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha 
            self.tree_ptr = (self.tree_ptr + 1) % self.capacity 

    def sample_batch(self, beta):
        
        assert len(self) >= self.batch_size
        assert beta >0
        idxs = self._sample_proportional()
        
        weights = np.array([self._calculate_weight(i, beta) for i in idxs])
        
        state = {}
        state2 = {}
        for key in self.state_buf.keys():
            state[key] = self.state_buf[key][idxs].copy()
            state2[key] = self.state2_buf[key][idxs].copy()
            
        batch = dict(state=state,
                     state2=state2,
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     weights=weights,
                     idxs=idxs)
        return batch

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert(priority) > 0, print(priority)
            assert 0 <= idx < len(self)
            idx = int(idx) 
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self)-1)
        segment = p_total / self.batch_size 
        
        for i in range(self.batch_size):
            a = segment*i 
            b = segment*(i+1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight


class SingleAgentReplyBuffer:
    def __init__(self, capacity, batch_size, state_shape, device, n_step=1):
        # init 
        self.capacity = capacity
        self.batch_size = batch_size
        local_shape = state_shape['local_shape']
        matrix_shape = state_shape['matrix_shape']
        global_shape = state_shape['global_shape']
        self.state_buf = {}
        self.state2_buf = {}
        self.state_buf['obs_info'] = np.empty((capacity, *local_shape), dtype=np.float32)
        self.state_buf['matrix_info'] = np.empty((capacity, matrix_shape), dtype=np.float32)
        self.state_buf['global_info'] = np.empty((capacity, *global_shape), dtype=np.float32)
        self.state_buf['mask_info'] = np.empty((capacity, local_shape[0]), dtype=np.float32)
        self.state2_buf['obs_info'] = np.empty((capacity, *local_shape), dtype=np.float32)
        self.state2_buf['matrix_info'] = np.empty((capacity, matrix_shape), dtype=np.float32)
        self.state2_buf['global_info'] = np.empty((capacity, *global_shape), dtype=np.float32)
        self.state2_buf['mask_info'] = np.empty((capacity, local_shape[0]), dtype=np.float32)
        
        self.act_buf = np.empty((capacity), dtype=np.float32)
        self.rew_buf = np.empty((capacity), dtype=np.float32)
        self.done_buf = np.empty((capacity), dtype=np.float32)
        self.mem_cntr = 0
        self.device = device

        # n_step learning 
        self.n_step_buf = deque(maxlen=3)
        self.n_step = n_step 
        self.gamma = 0.99

    def push(self, state, action, reward, state2, done):

        transition = (state, action, reward, state2, done)
        self.n_step_buf.append(transition)
        if len(self.n_step_buf) < self.n_step:
            return () 
        
        # make a n-step transition
        reward, state2, done = self._get_n_step_info(self.n_step_buf, self.gamma)
        state, action = self.n_step_buf[0][:2]

        #store
        idx = self.mem_cntr % self.capacity
        self.state_buf['obs_info'][idx] = state['obs_info']
        self.state_buf['matrix_info'][idx] = state['matrix_info']
        self.state_buf['global_info'][idx] = state['global_info']
        self.state_buf['mask_info'][idx] = state['mask_info']
        self.state2_buf['obs_info'][idx] = state2['obs_info']
        self.state2_buf['matrix_info'][idx] = state2['matrix_info']
        self.state2_buf['global_info'][idx] = state2['global_info']
        self.state2_buf['mask_info'][idx] = state['mask_info']
        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.done_buf[idx] = done
        self.mem_cntr +=1

        return self.n_step_buf[0]
    
    def sample_batch(self):
        size = min(self.mem_cntr, self.capacity)
        idxs = np.random.randint(0, size, size=self.batch_size)
        obs_info = self.state_buf['obs_info'][idxs]
        matrix_info = self.state_buf['matrix_info'][idxs]
        global_info = self.state_buf['global_info'][idxs]
        state = dict(obs_info=obs_info, matrix_info=matrix_info, global_info=global_info)
        obs_info = self.state2_buf['obs_info'][idxs]
        matrix_info = self.state2_buf['matrix_info'][idxs]
        global_info = self.state2_buf['global_info'][idxs]
        state2 = dict(obs_info=obs_info, matrix_info=matrix_info, global_info=global_info)
        batch = dict(state=state,
                     state2=state2,
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     idxs = idxs)
        return {k: T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                    for k,v in batch.items()}

    def sample_batch_from_idxs(self, idxs):

        # for n-step learning 
        batch =  dict(state=self.state_buf[idxs],
                     state2=self.state2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     )
        return {k: T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                    for k,v in batch.items()}

    def ready(self):
        if self.mem_cntr>=self.batch_size:
            return True
    
    def __len__(self):
            return min(self.mem_cntr, self.capacity)

    def _get_n_step_info(self, n_step_buffer, gamma):

        # info of the last transition 
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d  = transition[-3:]
            rew = r + gamma * rew * (1-d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        
        return rew, next_obs, done 


class PrioritizedReplayBuffer(SingleAgentReplyBuffer):
    """
    根据TD ERROR 的优先级来确定经验回放数组采样的优先级
    """
    def __init__(self, capacity, batch_size, state_shape, device, alpha, n_step=1):
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(capacity, batch_size, state_shape, device, n_step)
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        self.alpha = alpha
        while tree_capacity < self.capacity:
            tree_capacity *=2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def push(self, state, action, reward, state2, done):
        transition = super().push(state, action, reward, state2, done)
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha 
            self.tree_ptr = (self.tree_ptr + 1) % self.capacity 

    def sample_batch(self, beta):
        
        assert len(self) >= self.batch_size
        assert beta >0
        idxs = self._sample_proportional()
        
        weights = np.array([self._calculate_weight(i, beta) for i in idxs])
        obs_info = self.state_buf['obs_info'][idxs]
        matrix_info = self.state_buf['matrix_info'][idxs]
        global_info = self.state_buf['global_info'][idxs]
        mask_info = self.state_buf['mask_info'][idxs]
        state = dict(obs_info=obs_info, matrix_info=matrix_info, global_info=global_info, mask_info=mask_info)
        obs_info = self.state2_buf['obs_info'][idxs]
        matrix_info = self.state2_buf['matrix_info'][idxs]
        global_info = self.state2_buf['global_info'][idxs]
        mask_info = self.state2_buf['mask_info'][idxs]
        state2 = dict(obs_info=obs_info, matrix_info=matrix_info, global_info=global_info, mask_info=mask_info)
        batch = dict(state=state,
                     state2=state2,
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     weights=weights,
                     idxs=idxs)
        return batch

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert(priority) > 0
            assert 0 <= idx < len(self)
            idx = int(idx) 
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
        
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self)-1)
        segment = p_total / self.batch_size 
        
        for i in range(self.batch_size):
            a = segment*i 
            b = segment*(i+1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
